# Done: search the cheetah directory and print the names of the dirs containing histograms (cheetah style)
# Make a gui with a drop down list to load ??? *.cxi files? Nope just histogram files for now...

import fnmatch, os, sys
from PyQt4.QtGui import QDialog
from PyQt4.QtGui import QMainWindow
import PyQt4.uic
from PyQt4.QtGui import QApplication

f = open('Ui_gui.py', 'w')
PyQt4.uic.compileUi('gui.ui', f)
f.close()

from Ui_gui import Ui_MainWindow


cheetahdir   = '/home/amorgan/Desktop/nfs_home/data/fraglo/cheetah/'
h5_hist_path = 'data/data'
h5_hist_fnam = '*-histogram.h5'

hist_fnams = []
hist_dirs  = []
for dirname, dirnames, filenames in os.walk(cheetahdir):
    for filename in fnmatch.filter(filenames, h5_hist_fnam):
        hist_fnams.append(filename)
        hist_dirs.append(dirname)


def config_writer(h5dir, h5fnam, init=True):
    """
    Write a config file for MaxL_hist and put it into h5dir
    """
    pass


class MainWindow(QMainWindow):
    def __init__(self, hist_dirs, hist_fnams):
        QMainWindow.__init__(self)
        
        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # fill the histogram table 
        for m, hfnam in enumerate(hist_fnams):
            self.ui.tableWidget.insertRow(m)
            self.ui.tableWidget.setItem(m, 0, PyQt4.QtGui.QTableWidgetItem(hfnam))
        
        self.ui.tableWidget.resizeColumnsToContents()
        self.ui.tableWidget.resizeRowsToContents()
        
        self.resize(self.sizeHint())

if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = MainWindow(hist_dirs, hist_fnams)
    
    window.show()
    sys.exit(app.exec_())

