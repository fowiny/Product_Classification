import pysftp

# https://pysftp.readthedocs.io/en/release_0.2.8/pysftp.html

sftpConnect = pysftp.Connection(host="xxxxxxxx", username="rrrrr", private_key="wwwwwww", password=None, port=22, private_key_pass="uuuu", ciphers=None, log=False)

#sftpConnect.mkdir('productCategorization', mode=777) # Create a directory named recom with mode
#sftpConnect.rename('productCategorization', 'productCategorizer') # Rename directory from recom to recommend
#sftpConnect.rmdir('productCategorization') # Remove directory named recommend with mode

#### SAVING THE DIFFERENT OUTPUT FORMATS TO THE WEB SERVER RECOMMENDATION FOLDER USING single put_d() ###########
#sftpConnect.put_d('C:/Users/festo.owiny/Desktop/recommend/output/', '/home/converse/recommendationJson/', confirm=True, preserve_mtime=True)

#### SAVING THE DIFFERENT OUTPUT FORMATS TO THE WEB SERVER RECOMMENDATION FOLDER USING many put() ###########
sftpConnect.put('C:/Users/festo.owiny/Desktop/projects/productCategorization/output/productCategorized.xlsx', '/home/converse/productCategorization/productCategorized.xlsx')
sftpConnect.put('C:/Users/festo.owiny/Desktop/projects/productCategorization/output/Readme.txt', '/home/converse/productCategorization/Readme.txt')

print ('Does path /home/converse/productCategorization/ exisit ? :', sftpConnect.lexists('/home/converse/productCategorization/')) #Tests if path exisits
print ('Does file productCategorized.xlsx exist ? :', sftpConnect.lexists('/home/converse/productCategorization/productCategorized.xlsx')) #Tests if file exists
print ('Files on path /home/converse/productCategorization/ are:', sftpConnect.listdir('/home/converse/productCategorization/')) # List files in the given path

# Closes the connection
sftpConnect.close()
