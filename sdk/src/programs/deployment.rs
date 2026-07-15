// Copyright (C) 2019-2026 Provable Inc.
// This file is part of the Aleo SDK library.

// The Aleo SDK library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The Aleo SDK library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the Aleo SDK library. If not, see <https://www.gnu.org/licenses/>.

use crate::{
    types::{CertificateNative, DeploymentNative, VerifyingKeyNative},
    Address, Field, Program,
};

use pyo3::prelude::*;

use std::str::FromStr;

// Dummy verifying key + certificate for devnode deployments (the devnode
// skips certificate verification).  Copied from the wasm SDK's
// `buildDevnodeDeploymentTransaction`: the key encodes num_public_inputs=64
// to accommodate functions with many inputs/outputs.
const DEVNODE_VERIFIER_KEY: &str = "verifier1q9qqqqqqqqqqqqyvxgqqqqqqqqq87vsqqqqqqqqqhe7sqqqqqqqqqma4qqqqqqqqqq65yqqqqqqqqqqvqqqqqqqqqqqgtlaj49fmrk2d8slmselaj9tpucgxv6awu6yu4pfcn5xa0yy0tpxpc8wemasjvvxr9248vt3509vpk3u60ejyfd9xtvjmudpp7ljq2csk4yqz70ug3x8xp3xn3ul0yrrw0mvd2g8ju7rts50u3smue03gp99j88f0ky8h6fjlpvh58rmxv53mldmgrxa3fq6spsh8gt5whvsyu2rk4a2wmeyrgvvdf29pwp02srktxnvht3k6ff094usjtllggva2ym75xc4lzuqu9xx8ylfkm3qc7lf7ktk9uu9du5raukh828dzgq26hrarq5ajjl7pz7zk924kekjrp92r6jh9dpp05mxtuffwlmvew84dvnqrkre7lw29mkdzgdxwe7q8z0vnkv2vwwdraekw2va3plu7rkxhtnkuxvce0qkgxcxn5mtg9q2c3vxdf2r7jjse2g68dgvyh85q4mzfnvn07lletrpty3vypus00gfu9m47rzay4mh5w9f03z9zgzgzhkv0mupdqsk8naljqm9tc2qqzhf6yp3mnv2ey89xk7sw9pslzzlkndfd2upzmew4e4vnrkr556kexs9qrykkuhsr260mnrgh7uv0sp2meky0keeukaxgjdsnmy77kl48g3swcvqdjm50ejzr7x04vy7hn7anhd0xeetclxunnl7pd6e52qxdlr3nmutz4zr8f2xqa57a2zkl59a28w842cj4783zpy9hxw03k6vz4a3uu7sm072uqknpxjk8fyq4vxtqd08kd93c2mt40lj9ag35nm4rwcfjayejk57m9qqu83qnkrj3sz90pw808srmf705n2yu6gvqazpvu2mwm8x6mgtlsntxfhr0qas43rqxnccft36z4ygty86390t7vrt08derz8368z8ekn3yywxgp4uq24gm6e58tpp0lcvtpsm3nkwpnmzztx4qvkaf6vk38wg787h8mfpqqqqqqqqqqffkful";
const DEVNODE_CERTIFICATE: &str =
    "certificate1qyqsqqqqqqqqqqxvwszp09v860w62s2l4g6eqf0kzppyax5we36957ywqm2dplzwvvlqg0kwlnmhzfatnax7uaqt7yqqqw0sc4u";

/// A program deployment: the program, its synthesized verifying keys and
/// certificates, and (V9+) the program checksum and owner address.
///
/// Produced by `Process.deploy`; consumed by `Transaction.from_deployment`.
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Deployment(DeploymentNative);

#[pymethods]
impl Deployment {
    /// Constructs a Deployment from a JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> anyhow::Result<Self> {
        DeploymentNative::from_str(json).map(Self)
    }

    /// Builds a deployment WITHOUT synthesizing circuit keys: every function
    /// and record gets a shared dummy verifying key + certificate.  Devnodes
    /// accept it (they skip certificate verification); real networks reject
    /// it.  Mirrors the wasm SDK's `buildDevnodeDeploymentTransaction`
    /// (V9+ checksum/owner and V14+ record keys are always set).
    #[staticmethod]
    #[pyo3(signature = (program, owner, edition = 0))]
    fn from_program_unproven(
        program: &Program,
        owner: &Address,
        edition: u16,
    ) -> anyhow::Result<Self> {
        if program.functions().is_empty() {
            anyhow::bail!(
                "Attempted to create an empty deployment: {} has no functions",
                program.id()
            );
        }
        let vk = VerifyingKeyNative::from_str(DEVNODE_VERIFIER_KEY)?;
        let cert = CertificateNative::from_str(DEVNODE_CERTIFICATE)?;
        let mut verifying_keys =
            Vec::with_capacity(program.functions().len() + program.records().len());
        for function_name in program.functions().keys() {
            verifying_keys.push((*function_name, (vk.clone(), cert.clone())));
        }
        for record_name in program.records().keys() {
            verifying_keys.push((*record_name, (vk.clone(), cert.clone())));
        }
        let mut deployment =
            DeploymentNative::new(edition, (**program).clone(), verifying_keys, None, None)?;
        deployment.set_program_checksum_raw(Some(deployment.program().to_checksum()));
        deployment.set_program_owner_raw(Some(**owner));
        Ok(Self(deployment))
    }

    /// Returns the JSON string representation of the deployment.
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self.0)?)
    }

    /// Returns the deployment ID (the fee's `deployment_or_execution_id`).
    fn deployment_id(&self) -> anyhow::Result<Field> {
        self.0.to_deployment_id().map(Into::into)
    }

    /// Returns the program ID being deployed.
    fn program_id(&self) -> String {
        self.0.program_id().to_string()
    }

    /// Returns the number of functions in the deployed program.
    fn num_functions(&self) -> usize {
        self.0.program().functions().len()
    }

    fn __str__(&self) -> anyhow::Result<String> {
        self.to_json()
    }
}

impl From<DeploymentNative> for Deployment {
    fn from(value: DeploymentNative) -> Self {
        Self(value)
    }
}

impl From<Deployment> for DeploymentNative {
    fn from(value: Deployment) -> Self {
        value.0
    }
}

impl AsRef<DeploymentNative> for Deployment {
    fn as_ref(&self) -> &DeploymentNative {
        &self.0
    }
}
