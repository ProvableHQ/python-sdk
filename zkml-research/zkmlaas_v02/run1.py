import aleo
from helper import *

program_string = decision_tree_program_even_odd

program = aleo.Program.from_source(program_string)

private_key = aleo.PrivateKey()
destination = aleo.Account().address()
amount = aleo.Credits(0.3)
query = aleo.Query.rest("https://explorer.hamp.app")
process = aleo.Process.load()

process.add_program(program)
transfer_name = aleo.Identifier.from_string("main")
transfer_auth = process.authorize(private_key, program.id(), transfer_name, [
    aleo.Value.parse("{x0: 2i64, x1: 13i64}"),
    aleo.Value.parse("{x0: 2i64, x1: 13i64}"),
    aleo.Value.parse("{x0: 2i64, x1: 13i64}"),
    aleo.Value.parse("{x0: 2i64, x1: 13i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
    aleo.Value.parse("{x0: 2i64}"),
])
(_transfer_resp, transfer_trace) = process.execute(transfer_auth)
transfer_trace.prepare(query)
transfer_execution = transfer_trace.prove_execution(
    aleo.Locator(program.id(), aleo.Identifier.from_string("transfer")))
execution_id = transfer_execution.execution_id()
process.verify_execution(transfer_execution)

(fee_cost, _) = process.execution_cost(transfer_execution)
fee_priority = None
fee_auth = process.authorize_fee_public(
    private_key, fee_cost, execution_id, fee_priority)
(_fee_resp, fee_trace) = process.execute(fee_auth)
fee_trace.prepare(query)
fee = fee_trace.prove_fee()
process.verify_fee(fee, execution_id)

transaction = aleo.Transaction.from_execution(transfer_execution, fee)
a = transaction.to_json()

print(a)