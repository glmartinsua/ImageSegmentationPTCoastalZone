import pathlib

'''
name = 'FileName'

file = open("ConvertedNames.txt", "a")

i = 112
for path in pathlib.Path("blabla").iterdir():
    if path.is_file():
        if path.stem == 'ConvertedNames':
            continue
        if path.stem == 'namechanger':
            continue
        old_name = path.stem
        old_extension = path.suffix
        directory = path.parent
        new_name = name + str(i) + old_extension
        file.write(new_name + ' = ' + old_name + old_extension + '\n')
        path.rename(pathlib.Path(directory, new_name))
        i += 1

file.close()
'''