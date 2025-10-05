from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/api/parse', methods=['POST'])
def parse_solidity():
    """
    Endpoint to receive Solidity code and return CFG data.
    The actual parsing will be done on the frontend using @solidity-parser/parser.
    This endpoint can be used for additional processing if needed in the future.
    """
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        return jsonify({
            "status": "success",
            "message": "Parsing will be handled on the frontend with @solidity-parser/parser"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
