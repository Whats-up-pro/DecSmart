# Security Analysis Feature

## Tổng quan

Tính năng phân tích bảo mật tự động phát hiện các lỗ hổng phổ biến trong smart contract Solidity bằng cách kết hợp:
- **AST (Abstract Syntax Tree) Analysis**: Phân tích cấu trúc cú pháp
- **CFG (Control Flow Graph) Analysis**: Phân tích luồng điều khiển
- **Pattern Matching**: Tìm kiếm các pattern code nguy hiểm

## Các lỗ hổng được phát hiện

### 1. **Reentrancy** (Critical)
- External call trước khi cập nhật state
- Dễ bị tấn công reentrancy
- **Khuyến nghị**: Sử dụng Checks-Effects-Interactions pattern hoặc ReentrancyGuard

### 2. **Unchecked External Call Return Values** (High)
- Không kiểm tra giá trị trả về từ `.call()`, `.send()`
- Có thể bỏ sót lỗi giao dịch
- **Khuyến nghị**: Luôn kiểm tra return value với `require(success, "...")`

### 3. **Integer Overflow/Underflow** (High)
- Áp dụng cho Solidity < 0.8.0
- Phép toán số học không an toàn
- **Khuyến nghị**: Sử dụng SafeMath hoặc nâng cấp lên Solidity ^0.8.0

### 4. **tx.origin Authentication** (Medium)
- Sử dụng `tx.origin` thay vì `msg.sender`
- Dễ bị tấn công phishing
- **Khuyến nghị**: Luôn dùng `msg.sender` cho authorization

### 5. **Unprotected Selfdestruct** (Critical)
- Hàm `selfdestruct` không có access control
- Ai cũng có thể phá hủy contract
- **Khuyến nghị**: Thêm modifier `onlyOwner` hoặc access control

### 6. **Dangerous Delegatecall** (High)
- `delegatecall` đến địa chỉ do user kiểm soát
- Có thể chiếm quyền điều khiển contract
- **Khuyến nghị**: Chỉ delegatecall đến địa chỉ trusted, sử dụng whitelist

### 7. **Timestamp Dependence** (Medium)
- Logic phụ thuộc vào `block.timestamp` hoặc `now`
- Miner có thể thao túng timestamp
- **Khuyến nghị**: Dùng `block.number` hoặc oracle service

### 8. **Uninitialized Storage Pointer** (High)
- Storage pointer không được khởi tạo
- Có thể ghi đè lên state variable quan trọng
- **Khuyến nghị**: Luôn khởi tạo hoặc dùng `memory`

### 9. **Missing Access Control** (High)
- Hàm public/external thay đổi state không có kiểm soát truy cập
- Ai cũng có thể gọi hàm quan trọng
- **Khuyến nghị**: Thêm modifier hoặc `require(msg.sender == ...)`

### 10. **DoS with Block Gas Limit** (Medium)
- Loop chứa external call
- Có thể hết gas và block execution
- **Khuyến nghị**: Dùng pull over push pattern

### 11. **Unreachable Code** (Info)
- Code không bao giờ được thực thi
- Phát hiện qua CFG analysis
- **Khuyến nghị**: Xóa dead code hoặc sửa control flow

### 12. **Potential Infinite Loop** (Medium)
- Loop có thể chạy vô hạn
- Phát hiện qua cycle detection trong CFG
- **Khuyến nghị**: Đảm bảo loop có exit condition hợp lý

## Cách sử dụng

### Backend API

```bash
# Khởi động backend
cd backend
python app.py
```

**Endpoint**: `POST /api/v1/analyze`

**Request**:
```json
{
  "code": "contract MyContract { ... }"
}
```

**Response**:
```json
{
  "vulnerabilities": [
    {
      "type": "Reentrancy",
      "severity": "critical",
      "line": 15,
      "description": "External call on line 15 followed by state change",
      "recommendation": "Use checks-effects-interactions pattern",
      "nodeId": "node_5"
    }
  ],
  "summary": {
    "total": 5,
    "bySeverity": {
      "critical": 1,
      "high": 2,
      "medium": 2,
      "low": 0,
      "info": 0
    },
    "byType": {
      "Reentrancy": 1,
      "Unchecked Call Return Value": 2
    }
  },
  "score": 65,
  "cfg": {
    "nodes": [...],
    "edges": [...]
  }
}
```

### Frontend UI

1. **Nhập Solidity code** vào editor
2. **Click "🔍 Security Analysis"** để phân tích
3. **Xem kết quả**:
   - Security Score (0-100)
   - Vulnerability Summary (theo severity)
   - Chi tiết từng vulnerability
   - CFG với nodes bị highlight

### Ý nghĩa Security Score

- **80-100**: Excellent ✅ (màu xanh)
- **50-79**: Medium ⚠️ (màu vàng)
- **0-49**: Poor ❌ (màu đỏ)

**Công thức tính**:
```
Score = 100 - (Critical×20 + High×10 + Medium×5 + Low×2 + Info×1)
```

## Highlight trên CFG

Các node có lỗ hổng sẽ được:
- **Border**: Màu theo severity level
- **Box shadow**: Hiệu ứng glow
- **MiniMap**: Highlight rõ ràng

**Màu severity**:
- 🔴 **Critical**: `#d32f2f`
- 🟠 **High**: `#f57c00`
- 🟡 **Medium**: `#fbc02d`
- 🟢 **Low**: `#7cb342`
- 🔵 **Info**: `#0288d1`

## Test với Contract mẫu

File `vulnerable_contract_examples.sol` chứa:
- **VulnerableContract**: Contract có 10+ lỗ hổng cố ý
- **SecureContract**: Phiên bản đã fix các lỗ hổng

Sử dụng để test tính năng phát hiện vulnerability.

## Architecture

```
┌─────────────────┐
│   Frontend      │
│   (React)       │
└────────┬────────┘
         │ POST /api/v1/analyze
         │
┌────────▼────────┐
│   Backend       │
│   (Flask)       │
└────────┬────────┘
         │
    ┌────▼─────┐
    │ Security │
    │ Analyzer │
    └────┬─────┘
         │
    ┌────▼─────┐──────┐
    │ Pattern  │ CFG  │
    │ Matching │ Scan │
    └──────────┴──────┘
```

## Tích hợp

### Python (Backend)
```python
from security_analyzer import SecurityAnalyzer

analyzer = SecurityAnalyzer()
result = analyzer.analyze(
    code=solidity_code,
    cfg_data=cfg_result
)

print(f"Score: {result['score']}")
print(f"Vulnerabilities: {len(result['vulnerabilities'])}")
```

### JavaScript (Frontend)
```javascript
const response = await fetch('/api/v1/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ code: solidityCode })
});

const analysis = await response.json();
console.log(`Security Score: ${analysis.score}`);
```

## Hạn chế

1. **False Positives**: Có thể báo lỗi nhầm do pattern matching đơn giản
2. **False Negatives**: Không phát hiện được tất cả các lỗ hổng phức tạp
3. **Context Limited**: Không hiểu business logic của contract
4. **Static Analysis Only**: Không test runtime behavior

## Best Practices

1. ✅ **Kết hợp nhiều công cụ**: Slither, Mythril, Echidna
2. ✅ **Manual review**: Đừng chỉ tin vào automated tools
3. ✅ **Test coverage**: Viết unit tests và integration tests
4. ✅ **Formal verification**: Dùng cho critical contracts
5. ✅ **Security audit**: Thuê auditor chuyên nghiệp trước mainnet

## Tài liệu tham khảo

- [SWC Registry](https://swcregistry.io/) - Smart Contract Weakness Classification
- [Solidity Security Considerations](https://docs.soliditylang.org/en/latest/security-considerations.html)
- [Consensys Smart Contract Best Practices](https://consensys.github.io/smart-contract-best-practices/)
- [OpenZeppelin Contracts](https://docs.openzeppelin.com/contracts/)

## Roadmap

- [ ] Thêm nhiều vulnerability patterns
- [ ] Machine learning để giảm false positives
- [ ] Tích hợp Slither/Mythril
- [ ] Export report PDF
- [ ] CI/CD integration
- [ ] VSCode extension

## License

MIT
