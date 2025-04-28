import json

# 노트북 파일 읽기
with open('asset-dynamics.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 셀 내용 추출
cells = notebook['cells']

# 코드 셀과 마크다운 셀 분리
code_cells = [cell for cell in cells if cell['cell_type'] == 'code']
markdown_cells = [cell for cell in cells if cell['cell_type'] == 'markdown']

# 각 셀의 내용을 파일로 저장
with open('code_content.txt', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(code_cells):
        source = ''.join(cell['source'])
        f.write(f"--- CODE CELL {i+1} ---\n")
        f.write(source)
        f.write("\n\n")

with open('markdown_content.txt', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(markdown_cells):
        source = ''.join(cell['source'])
        f.write(f"--- MARKDOWN CELL {i+1} ---\n")
        f.write(source)
        f.write("\n\n")

print("Notebook content extracted to code_content.txt and markdown_content.txt") 