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
        
        self.process_options = None
        
        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.statusbar.showMessage('Load a process.config file...')
        self.ui.actionLoad_process_file.triggered.connect(self.loadProcess)
        self.ui.actionRunMaxL.triggered.connect(self.runMaxL)

    def runMaxL(self):
        if self.process_options is None :
            self.ui.statusbar.showMessage('You need to load some histograms!')
            return
        
        # get the selected histogram 
        n = self.ui.tableWidget.currentRow()
        print 'current histdir and name:', self.hist_dirs[n], self.hist_fnams[n]

        # get the selected algorithm 
        fnam = PyQt4.QtGui.QFileDialog.getOpenFileName(\
                directory = os.path.dirname(self.process_file),\
                filter = "Python files (*.py)")
        self.alg_file  = str(fnam)
        self.ui.statusbar.showMessage('loaded algorithm:' + self.alg_file)
        
        # export MaxLhist_MPI
        from subprocess import call
        print os.path.abspath('.')
        print 'export PYTHONPATH=$PYTHONPATH:' + os.path.abspath('./')+'/'
        call(['export PYTHONPATH=$PYTHONPATH:' + os.path.abspath('./')+'/'], shell=True)
        
        # run the script
        runstr = 'mpirun -n 10 python ' + self.alg_file + ' ' + self.hist_dirs[n] +'/'+ self.hist_fnams[n] +' '+ self.process_options['h5_data_path'] + ' -o ' + self.hist_dirs[n] + '/'
        
        # update the status
        self.ui.tableWidget.setItem(n, 1, PyQt4.QtGui.QTableWidgetItem('running...'))
        self.ui.tableWidget.resizeColumnsToContents()
        
        print '\n', runstr
        #import subprocess
        #p = subprocess.Popen([runstr], stdout=subprocess.PIPE,\
        #                               stderr=subprocess.PIPE,\
        #                               shell=True)
        #out, err = p.communicate()
        #print out
        
        call([runstr], shell=True)
        print 'done...'
        
        # reload the table
        self.load_hist_table(self.process_options, self.ui.tableWidget)


    def loadProcess(self):
        try :
            with open('.gui', 'r') as f:
                fnam = f.read()
                print '.gui file:', fnam
        except :
            fnam = './'
        
        fnam = PyQt4.QtGui.QFileDialog.getOpenFileName(\
                directory = fnam,\
                filter = "Config files (*.config)")
        self.process_file    = str(fnam)
        
        # save the last selection for the next time this program is run
        if self.process_file is not None and self.process_file != '':
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
        output_match = 'maxL-r*-histogram.h5'
        
        print '\nLoading histogram file names...'
        hist_fnams = []
        hist_dirs  = []
        status  = []
        for dirname, dirnames, filenames in os.walk(process_options['h5_hist_path']):
            for filename in fnmatch.filter(filenames, process_options['h5_hist_fnam']):
                hist_fnams.append(filename)
                hist_dirs.append(dirname)
                print filename
            
                if fnmatch.filter(filenames, output_match) != []:
                    status.append('done')
                else :
                    status.append('----')
        
        print '\nPopulating table...'
        tableWidget.setRowCount(0)
        for m, hfnam in enumerate(hist_fnams):
            tableWidget.insertRow(m)
            tableWidget.setItem(m, 0, PyQt4.QtGui.QTableWidgetItem(hfnam))
            tableWidget.setItem(m, 1, PyQt4.QtGui.QTableWidgetItem(status[m]))
        
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

