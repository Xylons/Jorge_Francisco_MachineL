import chardet
import pandas as pd

with open('text.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large

f=pd.read_csv('text.csv', encoding=result['encoding'])

# f=pd.read_csv("attacksNew.csv",encoding = "utf-8")
keep_col = ['UserID','UUID','Version','TimeStemp','GyroscopeStat_x_MEAN','GyroscopeStat_z_MEAN','GyroscopeStat_COV_z_x','GyroscopeStat_COV_z_y','MagneticField_x_MEAN','MagneticField_z_MEAN','MagneticField_COV_z_x','MagneticField_COV_z_y','Pressure_MEAN','LinearAcceleration_COV_z_x','LinearAcceleration_COV_z_y','LinearAcceleration_x_MEAN','LinearAcceleration_z_MEAN']
new_f = f[keep_col]
new_f.to_csv("newFile.csv", index=False)