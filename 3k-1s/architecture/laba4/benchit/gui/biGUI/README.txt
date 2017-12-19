create GUI template with Glade Interface Designer (GTK GUI Designer, package: glade-3):
   -> gui.glade

convert *.glade to usable XML-File:
   - gtk-builder-convert gui.glade gui.xml

Python Requirements:
   - python (>= 2.0)
   - pygtk (detailed package list not known, yet)
   - python-numpy
   - python-matplotlib

run BenchIT-GUI:
   - ./BIGUI.py      (in Benchit root folder)      or
   - ./BIGUI.py -p /path/to/bi_root_folder

shortkeys:
   - STRG+S  -  save changes for edited files
   - STRG+W  -  close subtabs of Compile/Run tab
