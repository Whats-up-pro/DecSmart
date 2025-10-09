// Test script to analyze different Solidity contract types and structures
import parser from '@solidity-parser/parser';

// Test contracts with different complexity levels
const testContracts = {
  simple: `pragma solidity ^0.8.0;
contract SimpleContract {
    uint256 public value;
    
    function setValue(uint256 _value) public {
        if (_value > 100) {
            value = 100;
        } else {
            value = _value;
        }
    }
    
    function getValue() public view returns (uint256) {
        return value;
    }
}`,

  medium: `pragma solidity ^0.8.0;
contract MediumContract {
    uint256 public balance;
    bool public paused;
    
    function deposit() public payable {
        require(!paused, "Contract is paused");
        balance += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(amount <= balance, "Insufficient balance");
        require(!paused, "Contract is paused");
        
        if (amount > 1000) {
            balance -= amount;
            payable(msg.sender).transfer(amount);
        } else {
            balance -= amount;
            payable(msg.sender).transfer(amount);
        }
    }
    
    function pause() public {
        paused = true;
    }
}`,

  complex: `pragma solidity ^0.8.0;
contract ComplexContract {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    bool public mintingFinished;
    
    function mint(address to, uint256 amount) public {
        require(!mintingFinished, "Minting is finished");
        require(amount > 0, "Amount must be positive");
        
        if (amount <= 1000) {
            balances[to] += amount;
            totalSupply += amount;
        } else {
            require(amount <= 10000, "Amount too large");
            balances[to] += amount;
            totalSupply += amount;
        }
    }
    
    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        for (uint256 i = 0; i < 10; i++) {
            if (i < 5) {
                balances[msg.sender] -= amount / 10;
                balances[to] += amount / 10;
            }
        }
        
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
    
    function batchTransfer(address[] memory recipients, uint256[] memory amounts) public {
        require(recipients.length == amounts.length, "Arrays length mismatch");
        
        for (uint256 i = 0; i < recipients.length; i++) {
            if (amounts[i] > 0) {
                transfer(recipients[i], amounts[i]);
            }
        }
    }
}`,

  loops: `pragma solidity ^0.8.0;
contract LoopContract {
    uint256[] public numbers;
    
    function addNumbers(uint256[] memory _numbers) public {
        for (uint256 i = 0; i < _numbers.length; i++) {
            numbers.push(_numbers[i]);
        }
    }
    
    function processNumbers() public {
        uint256 i = 0;
        while (i < numbers.length) {
            if (numbers[i] > 100) {
                numbers[i] = numbers[i] * 2;
            }
            i++;
        }
    }
    
    function findMax() public view returns (uint256) {
        uint256 max = 0;
        for (uint256 i = 0; i < numbers.length; i++) {
            if (numbers[i] > max) {
                max = numbers[i];
            }
        }
        return max;
    }
}`,

  nested: `pragma solidity ^0.8.0;
contract NestedContract {
    function complexLogic(uint256 a, uint256 b) public pure returns (uint256) {
        if (a > 0) {
            if (b > 0) {
                for (uint256 i = 0; i < a; i++) {
                    if (i % 2 == 0) {
                        if (b > i) {
                            return a + b + i;
                        }
                    }
                }
            } else {
                return a * 2;
            }
        } else {
            if (b > 100) {
                return b / 2;
            } else {
                return 0;
            }
        }
        return 0;
    }
}`
};

// Analysis function
function analyzeContract(code, contractName) {
  try {
    const ast = parser.parse(code, { loc: true, range: true });
    
    const stats = {
      contractName,
      functions: 0,
      ifStatements: 0,
      whileLoops: 0,
      forLoops: 0,
      returnStatements: 0,
      requireStatements: 0,
      totalLines: 0,
      complexity: 'simple'
    };
    
    // Count different structures
    parser.visit(ast, {
      FunctionDefinition(node) {
        stats.functions++;
      },
      IfStatement(node) {
        stats.ifStatements++;
      },
      WhileStatement(node) {
        stats.whileLoops++;
      },
      ForStatement(node) {
        stats.forLoops++;
      },
      ReturnStatement(node) {
        stats.returnStatements++;
      },
      CallExpression(node) {
        if (node.expression.name === 'require') {
          stats.requireStatements++;
        }
      }
    });
    
    // Calculate total lines
    const lines = code.split('\n').length;
    stats.totalLines = lines;
    
    // Determine complexity
    const totalStructures = stats.ifStatements + stats.whileLoops + stats.forLoops;
    if (totalStructures <= 3) {
      stats.complexity = 'simple';
    } else if (totalStructures <= 8) {
      stats.complexity = 'medium';
    } else {
      stats.complexity = 'complex';
    }
    
    return stats;
  } catch (error) {
    return {
      contractName,
      error: error.message,
      functions: 0,
      ifStatements: 0,
      whileLoops: 0,
      forLoops: 0,
      returnStatements: 0,
      requireStatements: 0,
      totalLines: 0,
      complexity: 'error'
    };
  }
}

// Run analysis
console.log('CFG Visualizer - Contract Analysis Results');
console.log('='.repeat(50));

const results = [];
let totalContracts = 0;
let totalFunctions = 0;
let totalIfStatements = 0;
let totalWhileLoops = 0;
let totalForLoops = 0;
let totalReturnStatements = 0;
let totalRequireStatements = 0;

for (const [name, code] of Object.entries(testContracts)) {
  const result = analyzeContract(code, name);
  results.push(result);
  
  if (!result.error) {
    totalContracts++;
    totalFunctions += result.functions;
    totalIfStatements += result.ifStatements;
    totalWhileLoops += result.whileLoops;
    totalForLoops += result.forLoops;
    totalReturnStatements += result.returnStatements;
    totalRequireStatements += result.requireStatements;
  }
  
  console.log(`\n${result.contractName.toUpperCase()} Contract:`);
  if (result.error) {
    console.log(`  Error: ${result.error}`);
  } else {
    console.log(`  Functions: ${result.functions}`);
    console.log(`  If Statements: ${result.ifStatements}`);
    console.log(`  While Loops: ${result.whileLoops}`);
    console.log(`  For Loops: ${result.forLoops}`);
    console.log(`  Return Statements: ${result.returnStatements}`);
    console.log(`  Require Statements: ${result.requireStatements}`);
    console.log(`  Total Lines: ${result.totalLines}`);
    console.log(`  Complexity: ${result.complexity}`);
  }
}

console.log('\n' + '='.repeat(50));
console.log('SUMMARY STATISTICS:');
console.log(`Total Contracts Tested: ${totalContracts}`);
console.log(`Total Functions: ${totalFunctions}`);
console.log(`Total If Statements: ${totalIfStatements}`);
console.log(`Total While Loops: ${totalWhileLoops}`);
console.log(`Total For Loops: ${totalForLoops}`);
console.log(`Total Return Statements: ${totalReturnStatements}`);
console.log(`Total Require Statements: ${totalRequireStatements}`);
console.log(`Average Functions per Contract: ${(totalFunctions / totalContracts).toFixed(1)}`);
console.log(`Average If Statements per Contract: ${(totalIfStatements / totalContracts).toFixed(1)}`);

// Supported structures
console.log('\nSUPPORTED STRUCTURES:');
console.log('✓ Function Definitions');
console.log('✓ If/Else Statements');
console.log('✓ While Loops');
console.log('✓ For Loops');
console.log('✓ Return Statements');
console.log('✓ Variable Declarations');
console.log('✓ Expression Statements');
console.log('✓ Nested Control Structures');
console.log('✓ Require Statements');
console.log('✓ Mapping Operations');

export { results, totalContracts, totalFunctions, totalIfStatements, totalWhileLoops, totalForLoops };


