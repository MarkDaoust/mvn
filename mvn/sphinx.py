import os 
import sys
import subprocess

def main(target = 'html',log = None ,errors = None):
    [dirname,filename] = os.path.split(__file__)    

    docPath = os.path.join(dirname,'doc')
    
    if log is None:
        log    = open(os.path.join(docPath,'errors.txt'),'w')
        
    if errors is None:    
        errors = open(os.path.join(docPath,   'log.txt'),'w')

    subprocess.Popen(
        ['make','clean'],
        cwd = docPath,
        stdout = log,
        stderr = errors,
    ).communicate()
    
    subprocess.Popen(
        ['make',target],
        cwd = docPath,
        stdout = log,
        stderr = errors,
    ).communicate()


if __name__ == '__main__':
    main(*sys.argv[1:], log = sys.stdin, errors = sys.stderr)
