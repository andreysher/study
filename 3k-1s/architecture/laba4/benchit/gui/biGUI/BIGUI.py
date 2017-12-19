#!/usr/bin/env python
 
# First run tutorial.glade through gtk-builder-convert with this command:
# gtk-builder-convert tutorial.glade tutorial.xml
# Then save this file as tutorial.py and make it executable using this command:
# chmod a+x tutorial.py
# And execute it:
# ./tutorial.py
 
import pygtk
pygtk.require("2.0")
import gtk

import sys
import os
import string
from numpy import array, zeros, nan, ma, isnan

import glib, subprocess, commands


from matplotlib.figure import Figure

# uncomment to select /GTK/GTKAgg/GTKCairo
#from matplotlib.backends.backend_gtk import FigureCanvasGTK as FigureCanvas
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas
#from matplotlib.backends.backend_gtkcairo import FigureCanvasGTKCairo as FigureCanvas

# or NavigationToolbar for classic
#from matplotlib.backends.backend_gtk import NavigationToolbar2GTK as NavigationToolbar
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NavigationToolbar



 
class BenchitGUI(object):
   #biRootDir = os.getcwd()
   biRootDir = None
   guiRootDir = None

   # ich weiss noch nicht wie ich wieder an die canvas und das ax (Axes) der canvas rankomme -> also erst mal globaler pointer :(
   ax = None
   canvas = None

   # leeres dictionary um sich die geoeffneten files / modificationsdaten der files zu speichern
   openedFiles = {}

   def __init__(self):
      # selber #################
      self.guiRootDir = os.path.dirname( sys.argv[0] )
      self.biRootDir = self.guiRootDir
      ##########################

      builder = gtk.Builder()
      builder.add_from_file( os.path.join(self.guiRootDir, 'gui.xml') )
      builder.connect_signals({ "on_BenchitGUI_destroy" : gtk.main_quit })
      self.window = builder.get_object("BenchitGUI")
      
      # selber #################
      
      # args durchsuchen
      try:
         i = sys.argv.index("-p")
      except:
         i = -1

      # benchit root dir setzen
      if i != -1:
         self.biRootDir = os.path.normpath(sys.argv[i + 1])

      # ein paar einstellung an den gui elementen machen
      self.setHostsTreeview(builder)
      self.setKernelTreeview(builder)
      self.setFilesTreeview(builder)
      self.setKernelNotebook(builder)
      self.setPlotNotebook(builder)
      self.setMenubar(builder)
      self.setSourceTextview(builder)
      self.setEnvTextview(builder)
      self.setBatchTextview(builder)
      self.setHWInfoTextview(builder)
      self.setPlainTextview(builder)
      self.setCompileAndRunButtons(builder)
      self.setHostsCheckButtons(builder)
      self.setCompileAndRunNotebook(builder)

      ##########################
      
      self.window.show()


   def setCompileAndRunNotebook(self, builder):
      #print "setCompileAndRunNotebook: ", self, builder
      notebook = builder.get_object("carNotebook")
      notebook.set_scrollable(True)
      notebook.connect("key-press-event", self.on_control_w)


   def on_control_w(self, notebook, event):
      #print "on_control_w: ", self, notebook, event, dictkey
      # die gedrueckte key / keycombi auslesen, fuehrt auf verschiedenen rechnern zu unterschiedlichen keys, zb Ctrl+W <-> Ctrl+Mod2+W, ABER event.keyval ist gleich
      #key = gtk.accelerator_get_label(event.keyval, event.state)

      # wenn die keycombi ctrl+w -> event.keyval=119 ist, dann schliesse das aktuelle tab des carNotebooks
      if event.keyval == 119:
         pagenum = notebook.get_current_page()
         notebook.remove_page(pagenum)


   def setHostsCheckButtons(self, builder):
      #print "setHostsCheckButtons: ", self, builder
      hosts = []
      treeview = builder.get_object("hostsTreeview")
      liststore = treeview.get_model()
      try:
         iterator = liststore.get_iter_root()
      except:
         iterator = None

      while iterator != None:
         hosts.append(liststore.get_value(iterator, 0))
         iterator = liststore.iter_next(iterator)

      # mal sehen wie das dann unter win gehen soll
      localhost = ''
      if sys.platform == 'linux2':
         localhost = commands.getoutput('hostname') 

      vbox = builder.get_object("vbox7")
      if len(hosts) > 0:
         for host in hosts:
            checkbutton = gtk.CheckButton(host)
            if localhost == host:
               checkbutton.set_active(True)
            #button.connect("toggled", self.callback, "check button 1")
            vbox.pack_start(checkbutton, True, True, 2)
            checkbutton.show()
      else:
         checkbutton = gtk.CheckButton('no hosts found')
         vbox.pack_start(checkbutton, True, True, 2)
         checkbutton.show()


   def setCompileAndRunButtons(self, builder):
      button = builder.get_object("compileButton")
      button.connect("clicked", self.on_compileKernel, builder)

      button = builder.get_object("runButton")
      button.connect("clicked", self.on_runKernel, builder)

      button = builder.get_object("carButton")
      button.connect("clicked", self.on_compileAndRunKernel, builder)


   def on_compileKernel(self, button, builder):
      #print "on_compileKernel: ", self, button, builder
      self.compileKernel(builder)


   def on_runKernel(self, button, builder):
      #print "on_runKernel: ", self, button, builder
      self.runKernel(builder)


   def on_compileAndRunKernel(self, button, builder):
      #print "on_compileAndRunKernel: ", self, button, builder
      self.compileAndRunKernel(builder)


   def compileAndRunKernel(self, builder):
      #print "compileAndRunKernel: ", self, builder
      # hole den kerneTreeview aus dem builder
      treeview = builder.get_object("kernelTreeview")
      # ermittle welches kernel grad angewaehlt ist
      path, tvcolumn = treeview.get_cursor()

      # hole die zweige des ermittelten kernels
      subdirs = self.getSubDirsFromTreeview(treeview, path)

      tooltip = ''
      for subdir in subdirs:
         tooltip = os.path.join(tooltip, subdir)

      directory = os.path.join('kernel', tooltip)

      binary = ''
      for subdir in subdirs:
         binary = '%s%s%s' % (binary, '.', subdir)
      # fuehrenden punkt loeschen und am ende ein .0 anfuegen, warum auch immer die kerne mit .0 enden
      binary = '%s%s' % (binary[1:], '.0')

      binary = os.path.join('bin', binary)

      label = '%s%s%s%s' % ('C&R: ', subdirs[0], '/../', subdirs[-1])

      toggledHosts = self.getToggledHosts(builder)
      if toggledHosts == []:
         print 'you have no hosts selected to run this command'

      # mal sehen wie das dann unter win gehen soll
      localhost = ''
      if sys.platform == 'linux2':
         localhost = commands.getoutput('hostname') 

      for host in toggledHosts:
         if host == localhost:
            # rufe die benchit compile.sh und run.sh mit dem selectierten kern
            compileCommand = '%s%s%s' % (os.path.join(self.biRootDir, 'COMPILE.SH'), ' ', directory)
            runCommand = '%s%s%s' % (os.path.join(self.biRootDir, 'RUN.SH'), ' ', binary)
            command = '%s%s%s' % (compileCommand, ' && ', runCommand)
            self.update_carNotebook(command, label, tooltip, builder)
         else:
            print 'remote measurement is actually not implemented'


   def compileKernel(self, builder):
      #print "compileKernel: ", self, builder
      # hole den kerneTreeview aus dem builder
      treeview = builder.get_object("kernelTreeview")
      # ermittle welches kernel grad angewaehlt ist
      path, tvcolumn = treeview.get_cursor()

      # hole die zweige des ermittelten kernels
      subdirs = self.getSubDirsFromTreeview(treeview, path)

      tooltip = ''
      for subdir in subdirs:
         tooltip = os.path.join(tooltip, subdir)

      directory = os.path.join('kernel', tooltip)

      label = '%s%s%s%s' % ('C: ', subdirs[0], '/../', subdirs[-1])

      toggledHosts = self.getToggledHosts(builder)
      if toggledHosts == []:
         print 'you have no hosts selected to run this command'

      # mal sehen wie das dann unter win gehen soll
      localhost = ''
      if sys.platform == 'linux2':
         localhost = commands.getoutput('hostname') 

      for host in toggledHosts:
         if host == localhost:
            # rufe die benchit compile.sh mit dem selectierten kern
            command = '%s%s%s' % (os.path.join(self.biRootDir, 'COMPILE.SH'), ' ', directory)
            self.update_carNotebook(command, label, tooltip, builder)
         else:
            print 'remote measurement is actually not implemented'


   def runKernel(self, builder):
      #print "compileKernel: ", self, builder
      # hole den kerneTreeview aus dem builder
      treeview = builder.get_object("kernelTreeview")
      # ermittle welches kernel grad angewaehlt ist
      path, tvcolumn = treeview.get_cursor()

      # hole die zweige des ermittelten kernels
      subdirs = self.getSubDirsFromTreeview(treeview, path)

      tooltip = ''
      for subdir in subdirs:
         tooltip = '%s%s%s' % (tooltip, '.', subdir)
      # fuehrenden punkt loeschen und am ende ein .0 anfuegen, warum auch immer die kerne mit .0 enden
      tooltip = '%s%s' % (tooltip[1:], '.0')

      binary = os.path.join('bin', tooltip)

      label = '%s%s%s%s%s' % ('R: ', subdirs[0], '..', subdirs[-1], '.0')

      toggledHosts = self.getToggledHosts(builder)
      if toggledHosts == []:
         print 'you have no hosts selected to run this command'

      # mal sehen wie das dann unter win gehen soll
      localhost = ''
      if sys.platform == 'linux2':
         localhost = commands.getoutput('hostname') 

      for host in toggledHosts:
         if host == localhost:
            # rufe die benchit run.sh mit dem selectierten kern
            command = '%s%s%s' % (os.path.join(self.biRootDir, 'RUN.SH'), ' ', binary)
            self.update_carNotebook(command, label, tooltip, builder)
         else:
            print 'remote measurement is actually not implemented'


   def getToggledHosts(self, builder):
      #print "getToggledHosts: ", self, builder
      vbox = builder.get_object("vbox7")
      hosts = []
      for checkbutton in vbox.get_children():
         if checkbutton.get_active():
            hosts.append( checkbutton.get_label() )

      # liefer die liste mit den host checkbuttons die momentan aktiviert sind
      return hosts


   def update_carNotebook(self, command, label, tooltip, builder):
      #print "update_carNotebook: ", self, command, builder
      notebook = builder.get_object("carNotebook")
      
      sw = gtk.ScrolledWindow()
      sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
      textview = gtk.TextView()
      textview.set_editable(False)
      textbuffer = textview.get_buffer()
      sw.add(textview)
      sw.show()
      textview.show()
      pagelabel = gtk.Label(label)
      pagelabel.set_tooltip_text(tooltip)
      notebook.append_page(sw, pagelabel)

      # starte das commando und uebergib einem io_watch den buffer in den er schreiben soll, wenn von dem compile/run command was auf stdout geschrieben wird
      proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      glib.io_add_watch(proc.stdout, glib.IO_IN | glib.IO_HUP, self.write_to_buffer_in_tab, textbuffer)
      

   # frag mich nciht, ich habs nur von http://pygabriel.wordpress.com/2009/07/27/redirecting-the-stdout-on-a-gtk-textview/
   def write_to_buffer_in_tab(self, filedescriptor, condition, textbuffer):
      if condition == glib.IO_IN or condition == glib.IO_IN | glib.IO_HUP:
         char = filedescriptor.read(1)
         textbuffer.insert_at_cursor(char)
         return True
      elif condition == glib.IO_HUP:
         return False
      else:
         return False


   def setMenubar(self, builder):
      imageMenuItem = builder.get_object("imagemenuitem11")
      imageMenuItem.connect("activate", self.showPrimNumbersWindow, builder)

      imageMenuItem = builder.get_object("imagemenuitem5")
      imageMenuItem.connect("activate", gtk.main_quit)

      
   def showPrimNumbersWindow(self, menuitem, builder):
      window = gtk.Window(gtk.WINDOW_TOPLEVEL)
      window.set_size_request(500, 500)
      window.set_title("Prim Numbers - Range 1 to 100000")

# das hier auskommentieren verursacht viellei datenlecks
#      window.connect("", lambda w: window.destroy())
#      window.connect("destroy", lambda w: window.destroy())

      vbox = gtk.VBox(False, 0)

      sw = gtk.ScrolledWindow()
      sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
      textview = gtk.TextView()
      textview.set_editable(False)
      textbuffer = textview.get_buffer()
      sw.add(textview)
      sw.show()
      textview.show()
      vbox.pack_start(sw)

      filepath = os.path.join(self.guiRootDir, 'primNumbers.txt')
      primNumberFile = open( filepath , 'r')

      if primNumberFile:
         string = primNumberFile.read()
         primNumberFile.close()
         textbuffer.set_text(string)

      window.add(vbox)
      vbox.show()
      window.show()


   def setKernelNotebook(self, builder):
      self.notebook = builder.get_object("kernelNotebook")
      self.notebook.connect('switch-page', self.on_pageOfKernelNotebook_selected, builder)


   def getSubDirsFromTreeview(self, treeview, path):
      #print "getSubDirsFromTreeview: ", self, treeview
      subdirs = []
      treestore = treeview.get_model()
      try:
         iterator = treestore.get_iter(path)
      except:
         iterator = None

      while iterator != None:
         subdirs.append(treestore.get_value(iterator, 0))
         iterator = treestore.iter_parent(iterator)

      # reihenfolge war falsch rum
      subdirs.reverse()
      # gebe die subdirectories im feld zurueck
      return subdirs      


   def on_pageOfKernelNotebook_selected(self, notebook, notneeded, pageNumber, builder):
      # print "on_pageOfKernelNotebook_selected: ", self, notebook, notneeded, pageNumber, builder
      # hole den kerneTreeview aus dem builder
      treeview = builder.get_object("kernelTreeview")
      # ermittle welches kernel grad angewaehlt ist
      path, tvcolumn = treeview.get_cursor()

      # hole die zweige des ermittelten kernels
      subdirs = self.getSubDirsFromTreeview(treeview, path)
      
      # ermittle das label der selectierten tabs
      page = notebook.get_nth_page(pageNumber)
      label = notebook.get_tab_label_text(page)

      # update den filesTreeview, jenachdem welches tab/page aktiviert ist
      self.updateKernelFilesTreeview( subdirs, label, builder )


   def setKernelTreeview(self, builder):
      self.treestore = gtk.TreeStore(str)

      kernelDir = os.path.join(self.biRootDir, 'kernel')
      c = kernelDir.count(os.sep)
      parentIter = [None]              # parents fuer die einzelnen TreeStores

      # erstelle den kernel baum
      for path, subdirs, files in os.walk(kernelDir):
         if c != path.count(os.sep):   # sonst wird kernel dir mit angezeigt
            tmpc = path.count(os.sep)
            if path.find('.svn') == -1:
               tmpIter = self.treestore.append(parentIter[tmpc - c - 1], [os.path.basename(path)])
               if tmpc - c == len(parentIter):
                  parentIter.append(tmpIter)
               else:
                  parentIter[tmpc - c] = tmpIter

      # create the TreeView using treestore
      self.treeview = builder.get_object("kernelTreeview")
      self.treeview.set_model(self.treestore)
      # create the TreeViewColumn to display the data
      self.tvcolumn = gtk.TreeViewColumn('Kernels')
      # add tvcolumn to treeview
      self.treeview.append_column(self.tvcolumn)
      # create a CellRendererText to render the data
      self.cell = gtk.CellRendererText()
      # add the cell to the tvcolumn and allow it to expand
      self.tvcolumn.pack_start(self.cell, True)
      # set the cell "text" attribute to column 0 - retrieve text
      # from that column in treestore
      self.tvcolumn.add_attribute(self.cell, 'text', 0)
      # make it NOT searchable
      self.treeview.set_enable_search(False)

#      self.treeview.set_show_expanders(False)
#      self.treeview.set_level_indentation(20)

      self.treeview.connect('row_activated', self.on_row_activated, builder)
      self.treeview.connect('row-expanded', self.on_row_expanded, builder)
      self.treeview.connect('row-collapsed', self.on_row_expanded, builder)


   def setFilesTreeview(self, builder):
      self.liststore = gtk.ListStore(str)
      # create the TreeView using treestore
      self.treeview = builder.get_object('kernelFilesTreeview')
      self.treeview.set_model(self.liststore)
      # create the TreeViewColumn to display the data
      self.tvcolumn = gtk.TreeViewColumn('Files')
      # add tvcolumn to treeview
      self.treeview.append_column(self.tvcolumn)
      # create a CellRendererText to render the data
      self.cell = gtk.CellRendererText()
      # add the cell to the tvcolumn and allow it to expand
      self.tvcolumn.pack_start(self.cell, True)
      # set the cell "text" attribute to column 0 - retrieve text
      # from that column in treestore
      self.tvcolumn.add_attribute(self.cell, 'text', 0)
      # make it NOT searchable
      self.treeview.set_enable_search(False)

      self.treeview.connect('row_activated', self.on_file_selected, builder)


   def on_file_selected(self, treeview, path, view_column, builder):
      #print "on_file_selected: ", self, treeview, path, view_column, builder
      # auslesen ob editTab, compileTab oder plotTab aktiviert ist
      self.notebook = builder.get_object('kernelNotebook')
      pagenum = self.notebook.get_current_page()
      page = self.notebook.get_nth_page(pagenum)
      label = self.notebook.get_tab_label_text(page)
      
      # je nachdem was angewaehlt ist, wird irgendwas gemacht ;)
      if label == 'Edit Source Code':
         self.on_sourcefile_selected(treeview, path, view_column, builder)
      elif label == 'Compile / Run':
         print 'mal sehen'
      elif label == 'Plot Result':
         self.on_plotfile_selected(treeview, path, view_column, builder)
      else:
         print 'Unknown label: ', label, ' in function: on_file_selected!'


   def on_sourcefile_selected(self, treeview, path, view_column, builder):
      #print "on_sourcefile_selected: ", self, treeview, path, view_column, builder
      # wie heisst das selectierte source file
      sourcefile = treeview.get_model().get_value( treeview.get_model().get_iter(path), 0 )
      
      # wie ist der pfad zu dem sourcefile
      treeview = builder.get_object("kernelTreeview")
      path, tvcolumn = treeview.get_cursor()

      # hole die zweige des ermittelten kernels
      subdirs = self.getSubDirsFromTreeview(treeview, path)
#      try:
#         iterator = treeview.get_model().get_iter(path)
#      except:
#         iterator = None
#
#      subdirs = []
#      while iterator != None:
#         subdirs.append(treeview.get_model().get_value(iterator, 0))
#         iterator = treeview.get_model().iter_parent(iterator)
#      # reihenfolge war falsch rum
#      subdirs.reverse()
      
      directory = os.path.join(self.biRootDir, 'kernel')
      for subdir in subdirs:
         directory = os.path.join(directory, subdir)
      
      # hole entsprechende textview
      sourceTextview = builder.get_object("sourceTextview")
      sourceFileBuffer = sourceTextview.get_buffer()

      # sourcefile einlesen und dem buffer uebergeben
      filepath = os.path.join(directory, sourcefile)
      sourceFile = open(filepath, 'r')
      if sourceFile:
         self.updateOpenedFiles('sourceTextview', filepath)
         sourceString = sourceFile.read()
         sourceFile.close()
      else:
         self.updateOpenedFiles('sourceTextview', '')
         sourceString = '%s%s%s' % ('File: ', filepath, ' not found')

      sourceFileBuffer.set_text(sourceString)


   def setSourceTextview(self, builder):
      #print "setSourceTextview: ", self, builder
      textview = builder.get_object('sourceTextview')
      textview.connect("key-press-event", self.on_control_s, 'sourceTextview')

   def setEnvTextview(self, builder):
      #print "setEnvTextview: ", self, builder
      textview = builder.get_object('envTextview')
      textview.connect("key-press-event", self.on_control_s, 'envTextview')

   def setBatchTextview(self, builder):
      #print "setBatchTextview: ", self, builder
      textview = builder.get_object('batchTextview')
      textview.connect("key-press-event", self.on_control_s, 'batchTextview')

   def setHWInfoTextview(self, builder):
      #print "setHWInfoTextview: ", self, builder
      textview = builder.get_object('hwInfoTextview')
      textview.connect("key-press-event", self.on_control_s, 'hwInfoTextview')

   def setPlainTextview(self, builder):
      #print "setPlainTextview: ", self, builder
      textview = builder.get_object('env_plainTextview')
      textview.connect("key-press-event", self.on_control_s_in_PlainTextview, 'plainTextview', builder)
      textview = builder.get_object('value_plainTextview')
      textview.connect("key-press-event", self.on_control_s_in_PlainTextview, 'plainTextview', builder)


   def updateOpenedFiles(self, string, filepath):
      #print "updateOpenedFiles: ", self, string, filepath
      if os.path.exists(filepath):
         self.openedFiles[string] = (filepath, os.path.getmtime(filepath))
      else:
         self.openedFiles[string] = ('', 0.0)


   def getOpenedFiles(self, dictkey):
      #print "getOpenedFiles: ", self, dictkey
      if dictkey in self.openedFiles:
         return self.openedFiles[dictkey]
      else:
         return ('', 0.0)


   def on_control_s(self, textview, event, dictkey):
      #print "on_control_s: ", self, textview, event, dictkey
      # die gedrueckte key / keycombi auslesen, fuehrt auf verschiedenen rechnern zu unterschiedlichen keys, zb Ctrl+S <-> Ctrl+Mod2+S, ABER event.keyval ist gleich
      #key = gtk.accelerator_get_label(event.keyval, event.state)

      # wenn die keycombi ctrl+s -> event.keyval=115 ist, dann speicher den textbuffer in die datei
      if event.keyval == 115:
         # hole den filenamen dessen inhalt im buffer steckt und hole den timestamp der letzten modifikation als die datei eingelesen wurde
         filepath, lastmod = self.getOpenedFiles(dictkey)
         if os.path.exists(filepath):
            if os.path.getmtime(filepath) == lastmod:

               # hole den text aus dem textbuffer des textview
               textbuffer = textview.get_buffer()
               text = textbuffer.get_text( textbuffer.get_start_iter(), textbuffer.get_end_iter(), True )

               # file hinter filepath oeffnen und inhalt des textbuffers reinschreiben         
               outfile = open(filepath, 'w')
               if outfile:
                  outfile.write(text)
                  outfile.close()
                  self.updateOpenedFiles(dictkey, filepath)
               else:
                  print '%s%s%s' % ('File: ', filepath, ' not found')

            else:
               print '%s%s%s' % ('File: ', filepath, ' has modified by another program')

         else:
            print '%s%s%s' % ('File: ', filepath, ' not found')
 

   def on_control_s_in_PlainTextview(self, textview, event, dictkey, builder):
      #print "on_control_s: ", self, textview, event, dictkey
      # die gedrueckte key / keycombi auslesen
      key = gtk.accelerator_get_label(event.keyval, event.state)

      # wenn die keycombi ctrl+s ist, dann speicher den textbuffer in die datei
      if key == 'Ctrl+S':
         # hole den filenamen dessen inhalt im buffer steckt und hole den timestamp der letzten modifikation als die datei eingelesen wurde
         filepath, lastmod = self.getOpenedFiles(dictkey)
         if os.path.exists(filepath):
            if os.path.getmtime(filepath) == lastmod:

               # hole den text aus dem textbuffer der 2 textviews
               textview = builder.get_object('env_plainTextview')
               textbuffer = textview.get_buffer()
               text1 = textbuffer.get_text( textbuffer.get_start_iter(), textbuffer.get_end_iter(), True )
               textview = builder.get_object('value_plainTextview')
               textbuffer = textview.get_buffer()
               text2 = textbuffer.get_text( textbuffer.get_start_iter(), textbuffer.get_end_iter(), True )

               # file hinter filepath oeffnen und inhalt des textbuffers reinschreiben         
               outfile = open(filepath, 'w')
               if outfile:
                  outfile.write(text1)
                  outfile.write(text2)
                  outfile.close()
                  self.updateOpenedFiles(dictkey, filepath)

                  # wenn die zahlen geaendert werden muss naturlich auch der plot geaendert werden
                  self.updateViewPlot(filepath, builder)
               else:
                  print '%s%s%s' % ('File: ', filepath, ' not found')

            else:
               print '%s%s%s' % ('File: ', filepath, ' has modified by another program')

         else:
            print '%s%s%s' % ('File: ', filepath, ' not found')
 

   def on_plotfile_selected(self, treeview, path, view_column, builder):
      #print "on_plotfile_selected: ", self, treeview, path, view_column, builder
      # wie heisst das selectierte result file
      resultfile = treeview.get_model().get_value( treeview.get_model().get_iter(path), 0 )
      
      # wie ist der pfad zu dem resultfile
      treeview = builder.get_object("kernelTreeview")
      path, tvcolumn = treeview.get_cursor()

      # hole die zweige des ermittelten kernels
      subdirs = self.getSubDirsFromTreeview(treeview, path)
#      iterator = treeview.get_model().get_iter(path)
#      subdirs = []
#      while iterator != None:
#         subdirs.append(treeview.get_model().get_value(iterator, 0))
#         iterator = treeview.get_model().iter_parent(iterator)
#      # reihenfolge war falsch rum
#      subdirs.reverse()
      
      directory = os.path.join(self.biRootDir, 'output')
      for subdir in subdirs:
         directory = os.path.join(directory, subdir)

      filepath = os.path.join(directory, resultfile)

      # auslesen welches aktiviert ist
      self.notebook = builder.get_object('plotNotebook')
      pagenum = self.notebook.get_current_page()
      page = self.notebook.get_nth_page(pagenum)
      label = self.notebook.get_tab_label_text(page)

      # update die einzelnen tabs
      # je nachdem was angewaehlt ist, wird zuerst diess tab als prioritaet erachtet
      if label == 'View Plot':
         self.updateViewPlot(filepath, builder)
         #self.updateConfigPlot(filepath, builder)
         #self.updateExportPlot(filepath, builder)
         self.updatePlainTextview(filepath, builder)
      elif label == 'Config Plot':
         #self.updateConfigPlot(filepath, builder)
         #self.updateExportPlot(filepath, builder)
         self.updatePlainTextview(filepath, builder)
         self.updateViewPlot(filepath, builder)
      elif label == 'Export Plot':
         #self.updateExportPlot(filepath, builder)
         #self.updateConfigPlot(filepath, builder)
         self.updatePlainTextview(filepath, builder)
         self.updateViewPlot(filepath, builder)
      elif label == 'Plain Source':
         self.updatePlainTextview(filepath, builder)
         #self.updateConfigPlot(filepath, builder)
         #self.updateExportPlot(filepath, builder)
         self.updateViewPlot(filepath, builder)
      else:
         print 'Unknown label: ', label, ' in function: on_plotfile_selected!'


   def setPlotNotebook(self, builder):
      #print "setPlotNotebook: ", self, builder
      fig = Figure(figsize=(10,10), dpi=100)
      self.ax = fig.add_subplot(111)

      self.canvas = FigureCanvas(fig)
      notebook = builder.get_object("plotNotebook")
      notebook.remove_page(0)
      notebook.insert_page(self.canvas, gtk.Label('View Plot'), 0)

      self.canvas.show()

      # setze wieder auf 1. tab, das remove und insert hat den focus verschoben
      notebook.set_current_page(0)

   
   def updateViewPlot(self, filepath, builder):
      #print "updatePlainTextview: ", self, filepath, builder
      # oeffne das resultfile, lese inhalt ein, schliesse file wieder
      resultFile = open(filepath, 'r')

      if resultFile:
         # alle zeilen der datei lesen
         res = resultFile.readlines()
         pos1 = res.index('beginofdata\n') + 1
         pos2 = res.index('endofdata\n')
         # ergebnisse (messdaten) in eine liste konvertieren
         data = map(lambda x: string.split(x, '\t')[0:-1], res[pos1:pos2])

         # dummys erstellt um spaeter zu fuellen
         numfunc = len(data[0]) - 1
         x = zeros(pos2-pos1)
         y = zeros((pos2-pos1, numfunc))

         # dummys fuellen
         i = -1
         for row in data:
            i = i + 1
            x[i] = float(row[0])
            j = -1
            for elem in row[1:]:
               j = j + 1
               try:
                  y[i, j] = float(elem)
               except:
                  y[i, j] = nan

         # alte grafik loeschen
         self.ax.clear()

         # neue funktionen der grafik hinzufuegen
         for i in range(0, numfunc):
            self.ax.plot(x, y[:,i], 'o')

         # grafik neu zeichen
         self.canvas.draw()

         resultFile.close()
      else:
         print '%s%s%s' % ('File: ', filepath, ' not found')


   def updatePlainTextview(self, filepath, builder):
      #print "updatePlainTextview: ", self, filepath, builder
      # hole textview fuer die sachen die nix mit die wirklichen funktionsergebnissen zu tun haben
      env_plainTextview = builder.get_object("env_plainTextview")
      env_plainTextviewBuffer = env_plainTextview.get_buffer()
      # hole textview fuer die wirklichen funktionsergebnisse
      value_plainTextview = builder.get_object("value_plainTextview")
      value_plainTextviewBuffer = value_plainTextview.get_buffer()
      
      # oeffne das resultfile, lese inhalt ein, schliesse file wieder
      resultFile = open(filepath, 'r')

      if resultFile:
         self.updateOpenedFiles('plainTextview', filepath)
         string = resultFile.read()
         
         # schreibe in envString bis 'beginofdata' im resultFile gefunden wird
         # dort weiter lesen und die messergebnisse auslesen         
         pos = string.find('%s%s' % ('beginofdata', os.linesep))

         envString = string[:pos]
         valueString = string[pos:]

         resultFile.close()
      else:
         self.updateOpenedFiles('plainTextview', '')
         envString = '%s%s%s' % ('File: ', filepath, ' not found')
         valueString = '%s%s%s' % ('File: ', filepath, ' not found')

      # setze die buffer auf die richtigen strings
      env_plainTextviewBuffer.set_text(envString)
      value_plainTextviewBuffer.set_text(valueString)


   def on_row_activated(self, treeview, path, view_column, builder):
      # print "on_row_activated: ", self, treeview, path, view_column, builder
      path, tvcolumn = treeview.get_cursor()
      if treeview.row_expanded(path):
         treeview.collapse_row(path)
      else:
         if treeview.get_model().iter_has_child(treeview.get_model().get_iter(path)):
            treeview.expand_row(path, False)
         else:
            # wenn ich ein blatt bin, dann zeige nur die files
            self.on_row_expanded(treeview, treeview.get_model().get_iter(path), path, builder)


   def on_row_expanded(self, treeview, iterator, path, builder):
      # print "on_row_expanded: ", self, treeview, iterator, path, builder
      subdirs = []
      while iterator != None:
         subdirs.append(treeview.get_model().get_value(iterator, 0))
         iterator = treeview.get_model().iter_parent(iterator)

      # reihenfolge war falsch rum
      subdirs.reverse()

      self.notebook = builder.get_object('kernelNotebook')
      pagenum = self.notebook.get_current_page()
      page = self.notebook.get_nth_page(pagenum)
      label = self.notebook.get_tab_label_text(page)

      self.updateKernelFilesTreeview( subdirs, label, builder )


   def updateKernelFilesTreeview( self, selectedPathInKernelTreeview, selectedKernelNotebookPageLabel, builder ):
      treeview = builder.get_object('kernelFilesTreeview')
      liststore = treeview.get_model()
      liststore.clear()

      subdirs = selectedPathInKernelTreeview
      label = selectedKernelNotebookPageLabel

      # fuer die dateien etweder im kernel oder im output ordner
      if label == 'Edit Source Code':
         directory = os.path.join(self.biRootDir, 'kernel')

         for subdir in subdirs:
            directory = os.path.join(directory, subdir)

         # durchsuche das LOCALDEF dir nach hosts
         for files in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, files)):
               liststore.append(['%s' % files])

      # fuer die binaries im bin ordner
      elif label == 'Compile / Run':
         directory = os.path.join(self.biRootDir, 'bin')

         for files in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, files)):
               rightFile = True
               for substr in subdirs:
                  if files.find(substr) == -1:
                     rightFile = False
                     break
               if rightFile:
                  liststore.append(['%s' % files])

      elif label == 'Plot Result':
         directory = os.path.join(self.biRootDir, 'output')

         for subdir in subdirs:
            directory = os.path.join(directory, subdir)

         # nur die durchsuchen wenn es den ordner wirklich gibt, muss bei output nicht der fall sein
         if os.path.exists(directory):
            # durchsuche das output dir nach bit files
            for files in os.listdir(directory):
               if os.path.isfile(os.path.join(directory, files)) and (not files.endswith('.bit.gp') and (not files.endswith('.bit.gui'))):
                  liststore.append(['%s' % files])

      else:
         print 'Unknown label: ', label, ' in function: updateKernelFilesTreeview!'

      
   def setHostsTreeview(self, builder):
      self.liststore = gtk.ListStore(str)

      # durchsuche das LOCALDEF dir nach hosts
      for hostfile in os.listdir(os.path.join(self.biRootDir, 'LOCALDEFS')):
         if (not hostfile.endswith('_input_architecture')) and (not hostfile.endswith('_input_display')) and (not hostfile.startswith('PROTOTYPE')) and (not hostfile.startswith('.svn')):
            self.liststore.append(['%s' % hostfile])

      # create the TreeView using treestore
      self.treeview = builder.get_object("hostsTreeview")
      self.treeview.set_model(self.liststore)
      # create the TreeViewColumn to display the data
      self.tvcolumn = gtk.TreeViewColumn('Hosts')
      # add tvcolumn to treeview
      self.treeview.append_column(self.tvcolumn)
      # create a CellRendererText to render the data
      self.cell = gtk.CellRendererText()
      # add the cell to the tvcolumn and allow it to expand
      self.tvcolumn.pack_start(self.cell, True)
      # set the cell "text" attribute to column 0 - retrieve text
      # from that column in treestore
      self.tvcolumn.add_attribute(self.cell, 'text', 0)
      # make it NOT searchable
      self.treeview.set_enable_search(False)
      # make it searchable
      #self.treeview.set_search_column(0)
      # Allow sorting on the column
      #self.tvcolumn.set_sort_column_id(0)
      # Allow drag and drop reordering of rows
      #self.treeview.set_reorderable(True)

      self.treeview.connect('row_activated', self.on_host_selected, builder)

   # reagiere wenn Host angewaehlt wird
   def on_host_selected(self, treeview, path, view_column, builder):
      # print "on_host_selected: ", self, treeview, path, view_column, builder
      host = treeview.get_model().get_value( treeview.get_model().get_iter(path), 0 )
      
      # setze die buffer um die files einzulesen
      envTextview = builder.get_object("envTextview")
      envFileBuffer = envTextview.get_buffer()

      batchTextview = builder.get_object("batchTextview")
      batchFileBuffer = batchTextview.get_buffer()
      
      hwInfoTextview = builder.get_object("hwInfoTextview")
      hwInfoFileBuffer = hwInfoTextview.get_buffer()

      # oeffne das hostFile, lese inhalt ein, schliesse file wieder
      filepath = os.path.join(self.biRootDir, 'LOCALDEFS')
      filepath = os.path.join(filepath, host)
      envFile = open(filepath, 'r')
      if envFile:
         self.updateOpenedFiles('envTextview', filepath)

         envString = envFile.read()

         # suche das zum host gehoerende batchsystemFile in dem hostFile
         envFile.seek(0, 0)
         for line in envFile:
            if line.find('BENCHIT_ENVIRONMENT=') != -1:
               batch = line.split('\"')
               break

         envFile.close()
      else:
         self.updateOpenedFiles('envTextview', '')
         envString = '%s%s%s' % ('File: ', filepath, ' not found')
         batch = ['', 'unknown (cause: envFile not found)']

      # oeffne das zum hostFile gehoerende hardwareInfoFile, lese inhalt ein, schliesse file wieder
      filepath = os.path.join(self.biRootDir, 'LOCALDEFS')
      filepath = os.path.join(filepath, '%s%s' % (host, '_input_architecture'))
      hwInfoFile = open(filepath, 'r')
      if hwInfoFile:
         self.updateOpenedFiles('hwInfoTextview', filepath)
         hwInfoString = hwInfoFile.read()
         hwInfoFile.close()
      else:
         self.updateOpenedFiles('hwInfoTextview', '')
         hwInfoString = '%s%s%s' % ('File: ', filepath, ' not found')

      # oeffne das zum hostFile gehoerende batchsystemFile, lese inhalt ein, schliesse file wieder
      filepath = os.path.join(self.biRootDir, 'tools')
      filepath = os.path.join(filepath, 'environments')
      filepath = os.path.join(filepath, batch[1])
      batchFile = open(filepath, 'r')
      if batchFile:
         self.updateOpenedFiles('batchTextview', filepath)
         batchString = batchFile.read()
         batchFile.close()
      else:
         self.updateOpenedFiles('batchTextview', '')
         batchString = '%s%s%s' % ('File: ', filepath, ' not found')

      # setze die buffer auf die richtigen strings
      envFileBuffer.set_text(envString)
      hwInfoFileBuffer.set_text(hwInfoString)
      batchFileBuffer.set_text(batchString)


if __name__ == "__main__":
  app = BenchitGUI()
  gtk.main()

