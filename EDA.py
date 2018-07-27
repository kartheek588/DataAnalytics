import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from IPython.core.display import HTML,display
from IPython.display import display_javascript

class EDA:
    """Exploratory Data Analysis"""
    def __init__(self, df, target=None):
        self.df = df
        self.target = target
    
    def printHtml(self, txt):
        display(HTML(txt))
    
 
    def make_cell(s):
        text = s.replace('\n','\\n').replace("\"", "\\\"").replace("'", "\\'")
        text2 = """var t_cell = IPython.notebook.get_selected_cell()
        t_cell.set_text('{}');
        var t_index = IPython.notebook.get_cells().indexOf(t_cell);
        IPython.notebook.to_code(t_index);
        IPython.notebook.get_cell(t_index).render();""".format(text)
        display_javascript(text2, raw=True)
        
    def execute_cell():
        display(Javascript('IPython.notebook.execute_cell_range(IPython.notebook.get_selected_index()+1, IPython.notebook.get_selected_index()+2)'))
        #IPython.notebook.execute_cells (index)
         
    def insert_file(filename):
        with open(filename, 'r') as content_file:
            content = content_file.read()
        make_cell(content)
    
    def printDataType(self,col):
        col_type='number'
        if self.df[col].dtype == 'object':
            col_type = 'object'
        print('Data Type: {}'.format(col_type))
    
    def describe(self, col):
        print ('Describe: \n')
        print(self.df[col].describe())
        print('\n')
        
        if self.df[col].dtype == 'object':
            print('Unique Values: \n')
            print('{}'.format(self.df[col].unique()))
            print('frequence of values \n')
            print(self.df[col].value_counts())
        else:
            #plt.hist(self.df[col],bins='auto')
            #convert pandas DataFrame object to numpy array and sort
            h = np.asarray(self.df[col].dropna())
            h = sorted(h)

            #use the scipy stats module to fit a normal distirbution with same mean and standard deviation
            fit = stats.norm.pdf(h, np.mean(h), np.std(h)) 
 
            #plot both series on the histogram
            plt.plot(h,fit,'-',linewidth = 2)
            plt.hist(h,normed=True,bins = 'auto')  
            plt.vlines(np.mean(h), ymin=0, ymax=np.max(h), linewidth=5, label = 'mean')
            plt.vlines(np.median(h), ymin=0, ymax=np.max(h), linewidth=2, color = 'red', label='median')
            plt.show() 
    
    def sampleRows(self,col):
        print('Top few rows\n')
        print(self.df[col].head())
        print('\n\nBottom few rows\n')
        print(self.df[col].tail())
    
    def missingValues(self, col):
        missing = self.df[col].isnull().values.any()
        print('Missing Values: {}'.format(missing))
        if missing:
             total = self.df[col].isnull().sum()
             percent = (self.df[col].isnull().sum()/self.df.isnull().count())*100
             print('Total missing: {}\nPercentage:{}'.format(total, percent))
    
    def skewness(self, col):
        
        skew = self.df[col].skew()
        kurtosis = self.df[col].kurt()
        print( 'excess kurtosis of normal distribution (should be 0): {}'.format( kurtosis ))
        print( 'skewness of normal distribution (should be 0): {}'.format( skew ))
        if skew > 0:
            self.printHtml('{} is <b>Right skewed</b> ==> {}'.format(col, skew))
        elif skew < 0 :
            self.printHtml('{} is <b>Left skewed</b> ==> {}'.format(col, skew))
        self.df[col].plot(kind='density')
    
    def outliers(self, col):
        self.df.boxplot(column=col, return_type='axes', figsize=(8,8))
        plt.text(x=0.74, y=self.df[col].quantile(0.75), s="3rd Quartile")
        plt.text(x=0.8, y=self.df[col].median(), s="Median")
        plt.text(x=0.75, y=self.df[col].quantile(0.25), s="1st Quartile")
        plt.text(x=0.9, y=self.df[col].min(), s="Min")
        plt.text(x=0.9, y=self.df[col].max(), s="Max")
        plt.text(x=0.7, y=self.df[col].quantile(0.50), s="IQR", rotation=90, size=25)
    
    def explore(self):
        #loop through each column
        i=1
        for col in self.df.columns:
            print('******************************************************************************************************************')
            print('******************************************************************************************************************')
            self.printHtml('    {}. COLUMN==> <b>{}</b>     '.format(i,col))
            print('******************************************************************************************************************')
            print('******************************************************************************************************************')
            
            print('--------------------------------------')  
            self.printDataType(col)
            
            print('\n--------------------------------------')    
            self.describe(col)
            if self.df[col].dtype != 'object':
                #skwed?
                print('\nSkwed?--------------------------------------')    
                self.skewness(col)
                
                #outliers    
                print('\nOutliers?--------------------------------------')    
                self.outliers(col)
            
            #sample rows
            print('\n--------------------------------------')    
            self.sampleRows(col)

            
            # missing values??
            print('\n--------------------------------------')    
            self.missingValues(col)
            
            print('\n\n')
            i+=1

#df = pd.read_csv('../../input/bureau.csv')            
#eda = EDA(df)
#eda.explore()
            