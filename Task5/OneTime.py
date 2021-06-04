from PIL import Image
import os, glob
import fnmatch

 
def batch_image(in_dir, out_dir):
    if not os.path.exists(out_dir):
        print(out_dir, 'is not existed.')
        os.mkdir(out_dir)
    
    if not os.path.exists(in_dir):
        print(in_dir, 'is not existed.')
        return -1
    
    for folders in glob.glob(in_dir+'/*'):
        # print(count)

        count = 0
        NoOfImages = (len(fnmatch.filter(os.listdir(folders), '*.pgm')))
        for images in glob.glob(folders+"/*.pgm"):
            print(folders)
            filepath, filename = os.path.split(images)
            print(filename)
            x = folders.split('\\')
            print(x)
            out_file = x[-1] +'_'+ filename[:-4] + '.jpg'



            if count < int(NoOfImages*0.8):
                data = out_dir+'\Train'
            else :
                data  = out_dir+"\Test"

            im = Image.open(images)
            new_path = os.path.join(data, out_file)  
            
            count = count + 1
            im.save(os.path.join(data, out_file)) 

             
                    

 

if __name__=='__main__':
    batch_image(r'G:\SBME\CV\Tasks\CV\Task5\koloRayyyyy7', r'G:\SBME\CV\Tasks\CV\Task5\dataSet')