# Done: search the cheetah directory and print the names of the dirs containing histograms (cheetah style)
# Make a gui with a drop down list to load ??? *.cxi files? Nope just histogram files for now...

import fnmatch, os, sys
from PyQt4.QtGui import QDialog
from PyQt4.QtGui import QMainWindow
import PyQt4.uic
from PyQt4.QtGui import QApplication
import ConfigParser
import h5py
import numpy as np
import MaxLhist_MPI
import pyqtgraph as pg

Ui_MainWindow, QMainWindow = PyQt4.uic.loadUiType('gui.ui')

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
        
        self.ui.tableWidget.setEditTriggers( PyQt4.QtGui.QTableWidget.NoEditTriggers )
        self.ui.tableWidget.itemDoubleClicked.connect( self.preview_results )

        self.hist_projWidget = self.ui.graphicsView
        self.hist_projWidget.setTitle('projected histogram')
        self.hist_projWidget.setLabel('bottom', text = 'adus')
        self.hist_projWidget.setLabel('left', text = 'frequency normalised')
        
        self.hists_Widget = self.ui.graphicsView_2
        self.hists_Widget.setTitle(title='pixel histograms')
        self.hists_Widget.setLabel('bottom', text = 'adus')
        self.hists_Widget.setLabel('left', text = 'frequency')
        
        self.hists_e_Widget = self.ui.graphicsView_3
        self.hists_e_Widget.setTitle(title='pixel errors', name = 'p_errors')
        self.hists_e_Widget.setLabel('bottom', text = 'index')
        self.hists_e_Widget.setLabel('left', text = 'log likelihood error')

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
        if os.environ.has_key('PYTHONPATH'):
            os.environ['PYTHONPATH'] += os.path.abspath('./')+'/'
        else :
            os.environ['PYTHONPATH'] = os.path.abspath('./')+'/'
        
        # run the script
        runstr = 'mpirun -n 4 python ' + self.alg_file + ' ' + self.hist_dirs[n] +'/'+ self.hist_fnams[n] +' '+ self.process_options['h5_data_path'] + ' -o ' + self.hist_dirs[n] + '/'
        
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

    def preview_results(self, QwidgetItem):
        # get the result's fnam
        n    = QwidgetItem.row()
        if self.status[n] == 'done' :
            self.ui.actionUpdate_gain_offsets.setEnabled(True)
            self.ui.actionUpdate_Xs.setEnabled(True)
            self.ui.actionUpdate_counts.setEnabled(True)
             
            fnam = self.hist_dirs[n] + '/maxL-'+self.hist_fnams[n]
            print '\nloading:', fnam 
            
            H = MaxLhist_MPI.Histograms(fnam_sub_h5 = fnam)
            
            self.show_preview(H)
        else :
            self.ui.actionUpdate_gain_offsets.setEnabled(False)
            self.ui.actionUpdate_Xs.setEnabled(False)
            self.ui.actionUpdate_counts.setEnabled(False)
            print '\nNo maxL data nothing to do...'
            self.clear_preview()

    def clear_preview(self):
        self.hist_projWidget.clear()
        self.hists_Widget.clear()
        self.hists_e_Widget.clear()

    def show_preview(self, H):
        """
        display:
        the projected histogram
        the fits to the proj hist
        """
        self.clear_preview()
        pixels       = H.pix_map['pix']
        pixels_valid = np.where(H.pix_map['valid'])[0]
         
        mus_name  = 'offset'
        gs_name   = 'gain'
        errors    = H.errors
        total_counts = np.sum(H.pix_map['hist_cor'][pixels_valid])
        hist_proj    = np.sum(H.pix_map['hist_cor'][pixels_valid], axis=0) / total_counts
        p_errors  = H.pix_map['e'][pixels_valid]
        m_sort    = np.argsort(p_errors)
        mus       = H.pix_map['mu']['v'][pixels_valid]
        gs        = H.pix_map['g']['v'][pixels_valid]
        hists     = H.pix_map['hist'][pixels_valid]
        hists_cor = H.pix_map['hist_cor'][pixels_valid]
        ns        = H.pix_map['n']['v'][pixels_valid]
        
        # show f and the mu values
        counts = np.sum(H.pix_map['hist'][pixels_valid], axis=-1)
        total_counts = np.sum(counts)
        
        fi   = lambda f: H.Xs['v'][i] * np.sum(ns[:, i] * counts) / float(total_counts)
        ftot = np.sum([fi(i) for i in range(H.Xs.shape[0])], axis=0) 
        
        # display
        self.hist_proj_plotitem = self.hist_projWidget.plot(fillLevel = 0.0, fillBrush = 0.7, stepMode = True)
        self.hist_proj_plotitem.setData(hist_proj + 1.0e-10)
        
        f_tot           = np.zeros_like(H.Xs['v'][0])
        for i in range(H.Xs.shape[0]):
            self.hist_projWidget.plot(y = fi(i) + 1.0e-10, pen=(i, len(H.Xs)+1), width = 10)
        
        self.hist_projWidget.plot(y = ftot + 1.0e-10, pen=(len(H.Xs), len(H.Xs)+1), width = 10)

        # now plot the histograms
        m      = 0
        title  = "histogram pixel " + str(m) + ' error ' + str(int(p_errors[m])) + ' offset {0:.1f}'.format(mus[m]) + ' inv. gain {0:.1f}'.format(gs[m])
        curve_his = self.hists_Widget.plot(hists[m], fillLevel = 0.0, fillBrush = 0.7, stepMode = False)
        curve_fit = self.hists_Widget.plot(H.hist_fit(H.pix_map[pixels_valid][m], H.Xs), pen = (0, 255, 0))
        self.hists_Widget.setXLink('f')
        def replot():
            m = hline.value()
            m = m_sort[m]
            title = "histogram pixel " + str(m) + ' error ' + str(int(p_errors[m])) + ' offset {0:.1f}'.format(mus[m]) + ' inv. gain {0:.1f}'.format(gs[m])
            self.hists_Widget.setTitle(title)
            curve_his.setData(hists[m])
            curve_fit.setData(H.hist_fit(H.pix_map[pixels_valid][m], H.Xs))
        
        # plot the pixel errors
        self.hists_e_Widget.plot(p_errors[m_sort], pen=(255, 255, 255))
        self.hists_e_Widget.setXLink('mus')

        hline = pg.InfiniteLine(angle=90, movable=True, bounds = [0, mus.shape[0]-1])
        hline.sigPositionChanged.connect(replot)
        self.hists_e_Widget.addItem(hline)


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
        
        #self.resize(self.sizeHint())
        self.hist_fnams = hist_fnams
        self.hist_dirs  = hist_dirs
        self.status     = status


if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = MainWindow()
    
    window.show()
    sys.exit(app.exec_())

