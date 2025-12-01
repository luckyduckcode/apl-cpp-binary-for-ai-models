from markdown import markdown
from weasyprint import HTML

# Convert Markdown to HTML and render to PDF
md_path = 'docs/research_paper.md'
html_path = '/tmp/research_paper.html'
pdf_path = 'docs/research_paper.pdf'

with open(md_path, 'r', encoding='utf-8') as f:
    md = f.read()

html = markdown(md, output_format='html5')
# Basic HTML template
html_doc = f"""
<html>
<head>
<meta charset='utf-8'>
<style>body{{font-family: 'Helvetica', Arial, sans-serif; padding: 40px;}}</style>
</head>
<body>
{html}
</body>
</html>
"""

HTML(string=html_doc).write_pdf(pdf_path)
print('Wrote', pdf_path)
