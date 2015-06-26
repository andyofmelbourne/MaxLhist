# Done: search the cheetah directory and print the names of the dirs containing histograms (cheetah style)
# Make a gui with a drop down list to load ??? *.cxi files? Nope just histogram files for now...

import fnmatch, os, sys
from PyQt4.QtGui import QDialog
from PyQt4.QtGui import QMainWindow
import PyQt4.uic
from PyQt4.QtGui import QApplication
import ConfigParser

Ui_MainWindow, QMainWindow = PyQt4.uic.loadUiType('gui.ui')



def config_writer(h5dir, h5fnam, init=True):
    """
    Write a config file for MaxL_hist and put it into h5dir
    """
    pass


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        
        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.ui.actionLoad_process_file.triggered.connect(self.selectFile)


    def selectFile(self):
        try :
            f = open('.gui', 'r')
            fnam = f.read(self.process_file)
        except :
            fnam = '.'
        finally :
            f.close()
        
        fnam = PyQt4.QtGui.QFileDialog.getOpenFileName(\
                directory = fnam,\
                filter = "Config files (*.config)")
        self.process_file    = str(fnam)
        
        # save the last selection for the next time this program is run
        with open('.gui', 'w') as f:
            f.write(self.process_file)
        
        self.process_options = self.load_process_file(self.process_file)
        self.load_hist_table(self.process_options, self.ui.tableWidget)
    
    def load_process_file(self, process_file):
        config = ConfigParser.SafeConfigParser()
        config.read(process_file)
        options = dict(config.items('dirs'))
        
        print '\nloading options...'
        for key in options.keys():
            print key, options[key]
        return options

    def load_hist_table(self, process_options, tableWidget):
        """
        fill the histogram table 
        """
        cheetahdir   = '/home/amorgan/Desktop/nfs_home/data/fraglo/cheetah/'
        h5_hist_path = 'data/data'
        h5_hist_fnam = '*-histogram.h5'
        
        print '\nLoading histogram file names...'
        hist_fnams = []
        hist_dirs  = []
        for dirname, dirnames, filenames in os.walk(process_options['h5_hist_path']):
            for filename in fnmatch.filter(filenames, process_options['h5_hist_fnam']):
                hist_fnams.append(filename)
                hist_dirs.append(dirname)
                print filename
        
        print '\nPopulating table...'
        for m, hfnam in enumerate(hist_fnams):
            tableWidget.insertRow(m)
            tableWidget.setItem(m, 0, PyQt4.QtGui.QTableWidgetItem(hfnam))
        
        tableWidget.resizeColumnsToContents()
        tableWidget.resizeRowsToContents()
        
        self.resize(self.sizeHint())
        self.hist_fnams = hist_fnams
        self.hist_dirs  = hist_dirs


if __name__ == "__main__":
    
    app    = QApplication(sys.argv)
    window = MainWindow()
    
    window.show()
    sys.exit(app.exec_())

