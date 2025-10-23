// SPDX-License-Identifier: MIT
pragma solidity ^0.7.6;

/**
 * Vulnerable Smart Contract for Testing Security Analysis
 * This contract contains INTENTIONAL vulnerabilities for educational purposes
 * DO NOT use in production!
 */

contract VulnerableContract {
    mapping(address => uint256) public balances;
    address public owner;
    
    // Vulnerability: Unprotected constructor
    constructor() {
        owner = msg.sender;
    }
    
    // Vulnerability #1: Reentrancy
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // External call BEFORE state change - REENTRANCY!
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        // State change after external call
        balances[msg.sender] -= amount;
    }
    
    // Vulnerability #2: Unchecked external call
    function sendFunds(address payable recipient, uint256 amount) public {
        // Return value not checked!
        recipient.call{value: amount}("");
    }
    
    // Vulnerability #3: tx.origin authentication
    function transferOwnership(address newOwner) public {
        // Using tx.origin instead of msg.sender
        require(tx.origin == owner, "Not owner");
        owner = newOwner;
    }
    
    // Vulnerability #4: Integer overflow (Solidity < 0.8.0)
    function addBalance(uint256 amount) public {
        // No SafeMath - can overflow!
        balances[msg.sender] += amount;
    }
    
    // Vulnerability #5: Timestamp dependence
    function timeBasedReward() public {
        // Relying on block.timestamp for logic
        if (block.timestamp % 2 == 0) {
            balances[msg.sender] += 100;
        }
    }
    
    // Vulnerability #6: Unprotected selfdestruct
    function destroy(address payable recipient) public {
        // No access control!
        selfdestruct(recipient);
    }
    
    // Vulnerability #7: Delegatecall to user-provided address
    function delegateCallExternal(address target, bytes memory data) public {
        // Dangerous delegatecall!
        (bool success, ) = target.delegatecall(data);
        require(success, "Delegatecall failed");
    }
    
    // Vulnerability #8: DoS with block gas limit
    function distributeRewards(address[] memory recipients) public {
        for (uint256 i = 0; i < recipients.length; i++) {
            // External call in loop - can cause DoS!
            (bool success, ) = recipients[i].call{value: 1 ether}("");
            require(success, "Transfer failed");
        }
    }
    
    // Vulnerability #9: Missing access control
    function setBalance(address user, uint256 amount) public {
        // Anyone can call this!
        balances[user] = amount;
    }
    
    // Vulnerability #10: Uninitialized storage pointer (Solidity < 0.5.0)
    // Note: This would be a compiler error in newer versions
    // struct Data { uint256 value; }
    // function uninitializedStorage() public {
    //     Data storage data; // Uninitialized!
    //     data.value = 100;
    // }
    
    // Helper function to receive Ether
    receive() external payable {
        balances[msg.sender] += msg.value;
    }
    
    fallback() external payable {
        balances[msg.sender] += msg.value;
    }
}

/**
 * Secure version with fixes
 */
contract SecureContract {
    mapping(address => uint256) public balances;
    address public owner;
    bool private locked;
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    modifier nonReentrant() {
        require(!locked, "No reentrancy");
        locked = true;
        _;
        locked = false;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    // Fix #1: Reentrancy protection
    function withdraw(uint256 amount) public nonReentrant {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // State change BEFORE external call
        balances[msg.sender] -= amount;
        
        // External call after state change
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
    
    // Fix #2: Check return value
    function sendFunds(address payable recipient, uint256 amount) public onlyOwner {
        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Transfer failed");
    }
    
    // Fix #3: Use msg.sender
    function transferOwnership(address newOwner) public onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
    
    // Fix #4: Solidity 0.8+ has built-in overflow protection
    function addBalance(uint256 amount) public {
        balances[msg.sender] += amount; // Safe in 0.8+
    }
    
    // Fix #5: Use block number or oracle
    function blockBasedReward() public {
        if (block.number % 100 == 0) {
            balances[msg.sender] += 100;
        }
    }
    
    // Fix #6: Protected selfdestruct
    function destroy(address payable recipient) public onlyOwner {
        selfdestruct(recipient);
    }
    
    // Fix #7: Whitelist for delegatecall
    mapping(address => bool) public trustedContracts;
    
    function delegateCallExternal(address target, bytes memory data) public onlyOwner {
        require(trustedContracts[target], "Untrusted contract");
        (bool success, ) = target.delegatecall(data);
        require(success, "Delegatecall failed");
    }
    
    // Fix #8: Pull over push pattern
    mapping(address => uint256) public pendingRewards;
    
    function setPendingRewards(address[] memory recipients, uint256 amount) public onlyOwner {
        for (uint256 i = 0; i < recipients.length; i++) {
            pendingRewards[recipients[i]] += amount;
        }
    }
    
    function claimReward() public nonReentrant {
        uint256 reward = pendingRewards[msg.sender];
        require(reward > 0, "No rewards");
        
        pendingRewards[msg.sender] = 0;
        (bool success, ) = msg.sender.call{value: reward}("");
        require(success, "Transfer failed");
    }
    
    // Fix #9: Access control
    function setBalance(address user, uint256 amount) public onlyOwner {
        balances[user] = amount;
    }
    
    receive() external payable {
        balances[msg.sender] += msg.value;
    }
}
