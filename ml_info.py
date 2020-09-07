
import platform

def print_info(verbose=True):
    
    print('ML & associated libraries info')
            

    if verbose:
        print('  Info on Python (v%s) packages:'%platform.python_version())

        for m in sorted(['numpy','scipy','tensorflow','onnx','onnxconverter-common','keras-onnx','mxnet','pydot','graphviz', 'torch', 'protobuf']):
            installed_ver = False
            try:
                exec('import %s'%m)
                if m == 'hdmf':
                    import hdmf._version
                    installed_ver = 'v%s'%hdmf._version.get_versions()['version']
                else:
                    installed_ver = 'v%s'%eval('%s.__version__'%m)
            except Exception as e:
                installed_ver = '???'
            print('    %s%s(installed: %s)'%(m, ' '*(20-len(m)), installed_ver))


    
def main(args=None):
    """Main"""

    '''if args is None:
        args = parse_arguments()'''

    print_info()


if __name__ == "__main__":
    main()
