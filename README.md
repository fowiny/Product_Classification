### Project with 3ASPorts to classify products
While NIKE brand had labels, the other brands such as HADDAD, JORDAN did not have labels.
Since the different brands were strongly related in their descriptions, It was possible to train our model with the labelled NIKE then make predictions for the other brands.


1. run.bat -> Run this file to execute the productCategorizer.py script
2. sftp.bat -> Run this file to execute the sftp.py script
3. productCategorizer.ipynb -> ipython notebook for the classification tasks
4. productCategorizer.py -> Python script corresponding to 'Always Available Portal'
5. sftp.py -> Python script to securely transfer output result using sftp to the web server
6. dataset -> dataset corresponding to SQL query embedded in the python script
7. ouput -> Results output in this folder as productCategorized.xlsx containing a list of products with their Category obtained
      productCategorized.xlsx    -> Excel file containing a list of products with their Category obtained by the algorithm
      productCategory        -> cell with the product category table
      accuracyTest        -> cell with a summary result of algorithm accuracy
      dictionaryCategory    -> dictionary of Category keys used in the result table
