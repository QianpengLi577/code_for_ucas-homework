import subprocess,os

def plot_as_emf(figure, **kwargs):

    inkscape_path = kwargs.get('inkscape', "E:\Inkscape\\bin\inkscape.exe")

    filepath = kwargs.get('filename', None)

    if filepath is not None:

        path, filename = os.path.split(filepath)

        filename, extension = os.path.splitext(filename)

        svg_filepath = os.path.join(path, filename+'.svg')

        emf_filepath = os.path.join(path, filename+'.emf')

        figure.savefig(svg_filepath, format='svg')

        os.system('inkscape'+' --export-type="emf" '+svg_filepath)

        print('inkscape'+' --export-type="emf" '+svg_filepath)
        # subprocess.call([inkscape_path, ' --export-type="emf"',svg_filepath])

        # os.remove(svg_filepath)