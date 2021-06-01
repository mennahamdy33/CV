
from PIL import Image
import os, glob
 
def batch_image(in_dir, out_dir):
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)
    
    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    count = 0
    for files in glob.glob(in_dir+'/*'):
        filepath, filename = os.path.split(files)
        
        out_file = filename[0:9] + '.jpg'
        # print(filepath,',',filename, ',', out_file)
        im = Image.open(files)
        new_path = os.path.join(out_dir, out_file)
        print(count, ',', new_path)
        count = count + 1
        im.save(os.path.join(out_dir, out_file))
        
 

def readAllData(self):
    os.system('mkdir selected')
    for i in range(1, 40):
        if i < 10:
                b = "yaleB0" + ("%d") % i
        else:
                b = "yaleB" + ("%d") % i
        print (b)

        try:
            os.chdir(b);
            print (os.getcwd());
            os.system("cp  `ls| grep " + b + "_P00A.0[012].E.[012]..pgm` ../selected/")
            os.system("cp  `ls| grep " + b + "_P00A+035E+15.pgm` ../selected/")
            
            os.system("ls| grep " + b + "_P00A.0[012].E.[012]..pgm | wc -l")
            os.chdir("..");
            print (os.getcwd());
        except:
            pass

if __name__=='__main__':
    batch_image('./data/validation/ped_examples', './batch')