// Copyright (C) 2019-2023 Aleo Systems Inc.
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

use snarkvm::circuit::network::AleoV0;
use snarkvm::console::network::MainnetV0;
use snarkvm::prelude::coinbase::{
    CoinbasePuzzle, CoinbaseVerifyingKey, EpochChallenge, ProverSolution,
};
use snarkvm::prelude::query::Query;
use snarkvm::prelude::store::helpers::memory::BlockMemory;
use snarkvm::prelude::transaction::Transaction;
use snarkvm::prelude::{
    Address, Authorization, Boolean, Ciphertext, ComputeKey, Deployment, Execution, Fee, Field, Group,
    Identifier, Literal, Locator, Plaintext, PrivateKey, Process, Program, ProgramID, ProvingKey,
    Record, Response, Scalar, Signature, Trace, Transition, Value, VerifyingKey, ViewKey, I128,
    I16, I32, I64, I8, U128, U16, U32, U64, U8,
};

// Account types
pub type AddressNative = Address<CurrentNetwork>;
pub type ComputeKeyNative = ComputeKey<CurrentNetwork>;
pub type PrivateKeyNative = PrivateKey<CurrentNetwork>;
pub type SignatureNative = Signature<CurrentNetwork>;
pub type ViewKeyNative = ViewKey<CurrentNetwork>;

// Algebraic types
pub type FieldNative = Field<CurrentNetwork>;
pub type GroupNative = Group<CurrentNetwork>;
pub type ScalarNative = Scalar<CurrentNetwork>;

// Integer types
pub type BooleanNative = Boolean<CurrentNetwork>;
pub type I8Native = I8<CurrentNetwork>;
pub type I16Native = I16<CurrentNetwork>;
pub type I32Native = I32<CurrentNetwork>;
pub type I64Native = I64<CurrentNetwork>;
pub type I128Native = I128<CurrentNetwork>;
pub type U8Native = U8<CurrentNetwork>;
pub type U16Native = U16<CurrentNetwork>;
pub type U32Native = U32<CurrentNetwork>;
pub type U64Native = U64<CurrentNetwork>;
pub type U128Native = U128<CurrentNetwork>;

// Network types
pub type CurrentNetwork = MainnetV0;
pub type CurrentAleo = AleoV0;

// Program Types
type CurrentBlockMemory = BlockMemory<CurrentNetwork>;
pub type AuthorizationNative = Authorization<CurrentNetwork>;
pub type DeploymentNative = Deployment<CurrentNetwork>;
pub type ExecutionNative = Execution<CurrentNetwork>;
pub type FeeNative = Fee<CurrentNetwork>;
pub type IdentifierNative = Identifier<CurrentNetwork>;
pub type LiteralNative = Literal<CurrentNetwork>;
pub type LocatorNative = Locator<CurrentNetwork>;
pub type ProcessNative = Process<CurrentNetwork>;
pub type ProgramIDNative = ProgramID<CurrentNetwork>;
pub type ProgramNative = Program<CurrentNetwork>;
pub type ProvingKeyNative = ProvingKey<CurrentNetwork>;
pub type QueryNative = Query<CurrentNetwork, CurrentBlockMemory>;
pub type ResponseNative = Response<CurrentNetwork>;
pub type TraceNative = Trace<CurrentNetwork>;
pub type TransactionNative = Transaction<CurrentNetwork>;
pub type TransitionNative = Transition<CurrentNetwork>;
pub type ValueNative = Value<CurrentNetwork>;
pub type VerifyingKeyNative = VerifyingKey<CurrentNetwork>;

// Record types
pub type CiphertextNative = Ciphertext<CurrentNetwork>;
pub type PlaintextNative = Plaintext<CurrentNetwork>;
pub type RecordCiphertextNative = Record<CurrentNetwork, CiphertextNative>;
pub type RecordPlaintextNative = Record<CurrentNetwork, PlaintextNative>;

// Coinbase types
pub type CoinbasePuzzleNative = CoinbasePuzzle<CurrentNetwork>;
pub type CoinbaseVerifyingKeyNative = CoinbaseVerifyingKey<CurrentNetwork>;
pub type EpochChallengeNative = EpochChallenge<CurrentNetwork>;
pub type ProverSolutionNative = ProverSolution<CurrentNetwork>;
