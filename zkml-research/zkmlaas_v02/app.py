from flask import Flask, request, jsonify
import json
import re
import aleo

app = Flask(__name__)

@app.route('/addProgram', methods=['POST'])
def add_program():
    try:
        # Extracting program_string from the request
        program_string = request.json.get('program_string')
        if not program_string:
            return jsonify(success=False, message="No program string provided"), 400

        # Try to create an Aleo program from the source
        program = aleo.Program.from_source(program_string)

        # Extracting program name using regular expression
        match = re.search(r'program\s+(\w+).aleo', program_string)
        if match:
            program_name = match.group(1)
        else:
            return jsonify(success=False, message="Program name not found in the string"), 400

        # Storing the program as JSON
        with open(f"{program_name}.json", 'w') as file:
            json.dump({'program_string': program_string}, file)

        return jsonify(success=True)

    except Exception as e:
        return jsonify(success=False, message=str(e)), 500





@app.route('/prove', methods=['POST'])
def prove():
    try:
        # Extract program_name and inputs from the request
        program_name = request.json.get('program_name')
        inputs = request.json.get('inputs')

        # Load the program string from the JSON file
        try:
            with open(f"{program_name}.json", 'r') as file:
                program_data = json.load(file)
                program_string = program_data['program_string']
        except FileNotFoundError:
            return jsonify(success=False, message="Program not found"), 404

        # Load the Aleo program from the source
        program = aleo.Program.from_source(program_string)

        # Initialize Aleo components
        private_key = aleo.PrivateKey()
        destination = aleo.Account().address()
        amount = aleo.Credits(0.3)
        query = aleo.Query.rest("https://explorer.hamp.app")
        process = aleo.Process.load()

        process.add_program(program)
        transfer_name = aleo.Identifier.from_string("main")

        # Process inputs
        try:
            # Check if inputs is a JSON array of max size 16
            if isinstance(inputs, list) and len(inputs) <= 16:
                # Convert each input to a string before parsing
                input_values = [aleo.Value.parse(str(input_str).replace("'","")) for input_str in inputs]
            else:
                # If inputs is not an array or too large, return an error
                return jsonify(success=False, message="Invalid input format or size"), 400
        except Exception as e:
            return jsonify(success=False, message="Input processing failed: " + str(e)), 500

        # Authorize the transfer
        transfer_auth = process.authorize(private_key, program.id(), transfer_name, input_values)

        # Execute the transfer
        (_transfer_resp, transfer_trace) = process.execute(transfer_auth)
        transfer_trace.prepare(query)
        transfer_execution = transfer_trace.prove_execution(
            aleo.Locator(program.id(), aleo.Identifier.from_string("transfer")))
        execution_id = transfer_execution.execution_id()
        process.verify_execution(transfer_execution)

        # Calculate and authorize fee
        (fee_cost, _) = process.execution_cost(transfer_execution)
        fee_priority = None
        fee_auth = process.authorize_fee_public(
            private_key, fee_cost, execution_id, fee_priority)
        (_fee_resp, fee_trace) = process.execute(fee_auth)
        fee_trace.prepare(query)
        fee = fee_trace.prove_fee()
        process.verify_fee(fee, execution_id)

        # Create the transaction
        transaction = aleo.Transaction.from_execution(transfer_execution, fee)

        # Return the transaction
        return jsonify(success=True, transaction=str(transaction))

    except Exception as e:
        return jsonify(success=False, message=str(e)), 500








if __name__ == '__main__':
    app.run(debug=True)
