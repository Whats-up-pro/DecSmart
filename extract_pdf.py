import fitz
doc = fitz.open('DecSmart_ICISN_2026.pdf')
text = ''.join([page.get_text() for page in doc])
with open('paper_content.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done - extracted', len(text), 'characters')
