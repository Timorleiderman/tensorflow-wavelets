
py VersionData\UpdateVersion.py

py -m build

py -m twine upload --repository pypi dist/*