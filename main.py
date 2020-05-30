import segTools as st
from pathlib import Path
import cv2 as cv
import sys

sys.setrecursionlimit(2084*2084)
src_dir = r'C:\Users\siddh\Documents\Nuclei segmentaion and Breast cancer detection Using BCH\breast-cancer-icpr-contest'

training_paths =Path('../breast-cancer-icpr-contest').glob('*/*/*.bmp')
csv_paths = Path('../breast-cancer-icpr-contest').glob('*/*/*.csv')

training_paths = sorted([x for x in training_paths])
csv_paths      = sorted([x for x in csv_paths])
print('lengths of img and csv files',len(training_paths), " ",len(csv_paths))


for i in range(len(training_paths)):
    
        im_path = training_paths[i]
        print('impath ' , im_path)
        
        m_cell_blobs = st.get_mcells(csv_paths[i]) 
        m_cell_means = st.get_means(m_cell_blobs)
        
        
        orimg = cv.imread(str(im_path))
        
        brh = st.blueRatioHistogram(orimg.copy())
        print('brh calculated')
        
        gbt = cv.threshold(brh,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        print('gbt:')
        
        
        opened = st.open_image(gbt)
        opened = cv.threshold(opened,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        print('opened')
        
        
        blobs = st.blob_detect(opened)
        means = st.get_means(blobs)
        print('calculated means and blobs :'+ str(len(blobs)))
        blob_img = st.get_blob_img(blobs,means,opened.copy())
        
  
                #making directories
        slide_number = str(im_path).split('\\')[-1][:6]
        p_path = im_path.parent.parent
        s_path = p_path / 'segmented'
        s_path.mkdir(exist_ok=True)
        m_path = s_path / 'mitotic'
        m_path.mkdir(exist_ok=True)
        nm_path = s_path / 'non_mitotic'
        nm_path.mkdir(exist_ok=True)
        d_path =  s_path / 'detected_mitotic'
        d_path.mkdir(exist_ok=True)
        
        
        cv.imwrite(str(s_path / 'blob_img.png'), blob_img)   
        cv.imwrite(str(s_path / 'gbt.png'), gbt)                
        cv.imwrite(str(s_path / 'opened.png'), opened)
        cv.imwrite(str(s_path / 'orimg.png'), orimg) 
        break
        padded_img = st.pad_image(orimg)
        cells, m_cells = st.segment(padded_img,means, m_cell_means)
        detected_mcells,nm_cells = st.classify_cells(blobs, m_cell_blobs)
        
        patient = slide_number
        
        for i,x in enumerate(detected_mcells):
            name = patient +'_d_mcell_'+str(i)+ '.bmp'
            cv.imwrite(str(d_path / name), cells[x])
            
        for i,x in enumerate(nm_cells):
            name =  patient +'_nm_cell_'+ str(i) +  '.bmp'
            cv.imwrite(str(nm_path / name), cells[x])        
        
        for i,m_cell in enumerate(m_cells):
            name =  patient +'_m_cell_'+ str(i) +  '.bmp'
            cv.imwrite(str(m_path / name), m_cell)        
        
        

        
        
        
        

    
