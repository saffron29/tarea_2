"""Ayuda mínima para escribir notebooks desde un script (una lista de celdas markdown y de código)."""
import nbformat


def notebook(celdas):
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for tipo, texto in celdas:
        texto = texto.strip("\n")
        if tipo == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(texto))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(texto))
    return nb


def md(texto):
    return ("md", texto)


def code(texto):
    return ("code", texto)
