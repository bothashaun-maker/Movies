"""Read the shipped one-cell notebook's embedded stages for runtime tests."""
import ast


def notebook_stages(notebook):
    cell=next(cell for cell in notebook['cells'] if cell.get('id')=='run')
    source=''.join(cell['source'])
    package=next(node for node in ast.parse(source).body if isinstance(node,ast.Assign)
                 and any(isinstance(target,ast.Name) and target.id=='_H3_PIPELINE_SOURCES' for target in node.targets))
    settings='\n'.join(source.splitlines()[:package.lineno-1])
    return {'settings':settings,**ast.literal_eval(package.value)}
