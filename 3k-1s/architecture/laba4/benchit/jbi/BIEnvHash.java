/********************************************************************
 * BenchIT - Performance Measurement for Scientific Applications
 * Contact: developer@benchit.org
 *
 * $Id: BIEnvHash.template.java 1 2009-09-11 12:26:19Z william $
 * $URL: svn+ssh://william@rupert.zih.tu-dresden.de/svn-base/benchit-root/BenchITv6/tools/BIEnvHash.template.java $
 * For license details see COPYING in the package base directory
 *******************************************************************/
import java.util.HashMap;
import java.util.Iterator;
import java.io.*;

public class BIEnvHash
{
   private static BIEnvHash instance = null;
   private HashMap table = null;
   private BIEnvHash()
   {
      bi_initTable();
      bi_fillTable();
   }
   public static BIEnvHash getInstance()
   {
      if ( instance == null ) instance = new BIEnvHash();
      return instance;
   }
   /** Puts a new key value mapping into the map and returns the
     * previously assigned value to the key, or null, if it's a
     * new mapping. */
   public String bi_setEnv( String key, String value )
   {
      return bi_put( key, value );
   }
   /** Puts a new key value mapping into the map and returns the
     * previously assigned value to the key, or null, if it's a
     * new mapping. */
   public String bi_put( String key, String value )
   {
      if ( table == null ) bi_initTable();
      Object ret = table.put( key, value );
      if ( ret == null ) return null;
      return (String)ret;
   }
   public String bi_getEnv( String key )
   {
      if (System.getenv(key)!=null) return System.getenv(key);
      return bi_get( key );
   }
   public String bi_get( String key )
   {
      if ( table == null ) return null;
      return (String)(table.get( key ));
   }
   public void bi_dumpTable()
   {
      if ( table == null ) return;
      printf( "\nHashtable dump of all known environment variables at compiletime:" );
      printf( "\n Key            | Value" );
      printf( "\n----------------------------------" );
      Iterator it = table.keySet().iterator();
      while ( it.hasNext() )
      {
         String key = (String)(it.next());
         String value = (String)(table.get( key ));
         printf( "\n " + key + " | " + value );
      }
      printf( "\n" + bi_size() + " entries in total.\n" );
      flush();
   }
   public void dumpToOutputBuffer( StringBuffer outputBuffer )
   {
      outputBuffer.append( "beginofenvironmentvariables\n" );
      Iterator it = table.keySet().iterator();
      while ( it.hasNext() )
      {
         String key = (String)(it.next());
         String value = (String)(table.get( key ));
         StringBuffer b = new StringBuffer();
         for ( int i = 0; i < value.length(); i++ )
         {
            if ( value.charAt( i ) == '"' ) {
               /* check equal number of escape chars->insert escape char */
               int cnt = 0;
               for ( int j = i - 1;  j >= 0; j-- ) {
                  if ( value.charAt( j ) == '\\' ) cnt++;
               }
               /* escape quote */
               if ( cnt % 2 == 0 ) {
                  b.append( '\\' );
               }
            }
            b.append( value.charAt( i ) );
         }
         value = b.toString();
         outputBuffer.append( key + "=\"" + value + "\"\n" );
      }
      outputBuffer.append( "endofenvironmentvariables\n" );
   }
   public int bi_size()
   {
      if ( table != null ) return table.size();
      return -1;
   }
   public void bi_initTable()
   {
      table = new HashMap( 1009 );
   }
   private void printf( String str )
   {
      System.out.print( str );
   }
   private void flush()
   {
      System.out.flush();
      System.err.flush();
   }
   /** Takes a PARAMETER file as argument and adds it's variables
     * to the environment HashMap. */
   public boolean bi_readParameterFile( String fileName )
   {
      boolean retval = false;
      File textFile = null;
      try {
         textFile = new File( fileName );
      }
      catch ( NullPointerException npe ) {
         return retval;
      }
      if ( textFile != null ) {
         if ( textFile.exists() ) {
            FileReader in;
            char ch = '\0';
            int size = (int)(textFile.length());
            int read = 0;
            char inData[] = new char[size + 1];
            try {
               in = new FileReader( textFile );
               in.read( inData, 0, size );
               in.close();
            }
            catch( IOException ex ) {
               return retval;
            } // scan inData for environment variables
            int scanned = 0;
            int it = 0;
            ch = inData[scanned];
            StringBuffer buffer = null;
            while ( scanned < size ) {
               buffer = new StringBuffer();
               // read a line
               while ( ch != '\n' ) {
                  if ( ch != '\r' ) {
                     buffer.append( ch );
                  }
                  scanned += 1;
                  ch = inData[scanned];
               }
               // trim leading and trailing white spaces
               String lineValue = buffer.toString().trim();
               // now check the line
               int eqIndex = -1;
               try {
                  eqIndex = lineValue.indexOf( "=" );
               }
               catch ( NullPointerException npe ) {
                  return retval;
               }
               // only check lines containing an equals sign, no $ sign,
               // and the start with a capital letter
               if ( eqIndex >= 0 && lineValue.indexOf( "$" ) < 0 &&
                  lineValue.substring( 0, 1 ).matches( "[A-Z]" ) ) {
                  String varname = null;
                  String varvalue = null;
                  try {
                     varname = lineValue.substring( 0 , eqIndex );
                     varvalue = lineValue.substring( eqIndex + 1 );
                  }
                  catch ( IndexOutOfBoundsException ioobe ) {
                  }
                  if ( ( varname != null ) && ( varvalue != null ) ) {
                     boolean st1=false, en1=false, st2=false, en2=false;
                     boolean st3=false, en3=false;
                     try {
                        st1=varvalue.startsWith( "'" );
                        en1=varvalue.endsWith( "'" );
                        st2=varvalue.startsWith( "(" );
                        en2=varvalue.endsWith( ")" );
                        st3=varvalue.startsWith( "\"" );
                        en3=varvalue.endsWith( "\"" );
                     }
                     catch ( NullPointerException npe ) {
                        return retval;
                     }
                     if ( ( ( st1 == en1 ) && ( ( st1 != st2 ) && ( st1!=st3 ) ) ) ||
                          ( ( st2 == en2 ) && ( ( st2 != st1 ) && ( st2!=st3 ) ) ) ||
                          ( ( st3 == en3 ) && ( ( st3 != st1 ) && ( st3!=st2 ) ) ) ||
                          ( ( st1 == en1 ) && ( st2 == en2 ) && ( st3 == en3 )
                          && ( st1 == st2 ) && ( st2 == st3 ) && ( st3 == false ) ) ) {
                        table.put( varname, varvalue );
                     }
                  }
               }
               scanned += 1;
               ch = inData[scanned];
               it++;
            }
         }
         else {
            System.err.println( "BenchIT: PARAMETER file \"" + fileName
               + "\" doesn't exist." );
            return retval;
         }
      }
      else {
         System.err.println( "BenchIT: PARAMETER file \"" + fileName
               + "\" not found." );
         return retval;
      }
      return true;
   }
   /* special case: generated from outside and code will be
      appended to this file */
   /** Fills the table with predefined content. */
   public void bi_fillTable()
   {

      bi_put("BENCHITROOT", "/home/andrey/study/architecture/laba4/benchit");
      bi_put("BENCHIT_ARCH_SHORT", "In_2");
      bi_put("BENCHIT_ARCH_SPEED", "1695M");
      bi_put("BENCHIT_CC", "gcc");
      bi_put("BENCHIT_CC_C_FLAGS", "");
      bi_put("BENCHIT_CC_C_FLAGS_HIGH", "-O0");
      bi_put("BENCHIT_CC_C_FLAGS_OMP", "");
      bi_put("BENCHIT_CC_C_FLAGS_STD", "-O2");
      bi_put("BENCHIT_CC_LD", "gcc");
      bi_put("BENCHIT_CC_L_FLAGS", "-lm");
      bi_put("BENCHIT_COMPILE", "gcc   -O0 -I. -I/home/andrey/study/architecture/laba4/benchit -I/home/andrey/study/architecture/laba4/benchit/tools  -DDEBUGLEVEL=0 -c /home/andrey/study/architecture/laba4/benchit/benchit.c /home/andrey/study/architecture/laba4/benchit/tools/stringlib.c /home/andrey/study/architecture/laba4/benchit/tools/output.c /home/andrey/study/architecture/laba4/benchit/tools/bitWriter.c /home/andrey/study/architecture/laba4/benchit/tools/gnuWriter.c /home/andrey/study/architecture/laba4/benchit/bienvhash.c");
      bi_put("BENCHIT_COMPILETIME_CC", "gcc");
      bi_put("BENCHIT_COMPILE_SH_IS_CALLED", "1");
      bi_put("BENCHIT_CPP_ACML", "");
      bi_put("BENCHIT_CPP_ATLAS", "");
      bi_put("BENCHIT_CPP_BLAS", "");
      bi_put("BENCHIT_CPP_ESSL", "");
      bi_put("BENCHIT_CPP_FFTW3", "");
      bi_put("BENCHIT_CPP_MKL", "");
      bi_put("BENCHIT_CPP_MPI", " -DUSE_MPI");
      bi_put("BENCHIT_CPP_PAPI", "-DUSE_PAPI");
      bi_put("BENCHIT_CPP_PCL", " -DUSE_PCL");
      bi_put("BENCHIT_CPP_PTHREADS", "");
      bi_put("BENCHIT_CPP_PVM", "");
      bi_put("BENCHIT_CPP_SCSL", "");
      bi_put("BENCHIT_CROSSCOMPILE", "0");
      bi_put("BENCHIT_CXX", "g++");
      bi_put("BENCHIT_CXX_C_FLAGS", "");
      bi_put("BENCHIT_CXX_C_FLAGS_HIGH", "-O3");
      bi_put("BENCHIT_CXX_C_FLAGS_OMP", "");
      bi_put("BENCHIT_CXX_C_FLAGS_STD", "-O2");
      bi_put("BENCHIT_CXX_LD", "g++");
      bi_put("BENCHIT_CXX_L_FLAGS", "-lm");
      bi_put("BENCHIT_DEBUGLEVEL", "0");
      bi_put("BENCHIT_DEFINES", " -DDEBUGLEVEL=0");
      bi_put("BENCHIT_ENVIRONMENT", "DEFAULT");
      bi_put("BENCHIT_F77", "gfortran");
      bi_put("BENCHIT_F77_C_FLAGS", "");
      bi_put("BENCHIT_F77_C_FLAGS_HIGH", "-O3");
      bi_put("BENCHIT_F77_C_FLAGS_OMP", "");
      bi_put("BENCHIT_F77_C_FLAGS_STD", "-O2");
      bi_put("BENCHIT_F77_LD", "gfortran");
      bi_put("BENCHIT_F77_L_FLAGS", "-lm");
      bi_put("BENCHIT_F90", "gfortran");
      bi_put("BENCHIT_F90_C_FLAGS", "");
      bi_put("BENCHIT_F90_C_FLAGS_HIGH", "-O3");
      bi_put("BENCHIT_F90_C_FLAGS_OMP", "");
      bi_put("BENCHIT_F90_C_FLAGS_STD", "-O2");
      bi_put("BENCHIT_F90_LD", "gfortran");
      bi_put("BENCHIT_F90_L_FLAGS", "-lm");
      bi_put("BENCHIT_F90_SOURCE_FORMAT_FLAG", "");
      bi_put("BENCHIT_F95", "gfortran");
      bi_put("BENCHIT_F95_C_FLAGS", "");
      bi_put("BENCHIT_F95_C_FLAGS_HIGH", "-O4");
      bi_put("BENCHIT_F95_C_FLAGS_OMP", "");
      bi_put("BENCHIT_F95_C_FLAGS_STD", "-O2");
      bi_put("BENCHIT_F95_LD", "gfortran");
      bi_put("BENCHIT_F95_L_FLAGS", "-lm");
      bi_put("BENCHIT_F95_SOURCE_FORMAT_FLAG", "");
      bi_put("BENCHIT_FILENAME_COMMENT", "0");
      bi_put("BENCHIT_FILES", "/home/andrey/study/architecture/laba4/benchit/benchit.c /home/andrey/study/architecture/laba4/benchit/tools/stringlib.c /home/andrey/study/architecture/laba4/benchit/tools/output.c /home/andrey/study/architecture/laba4/benchit/tools/bitWriter.c /home/andrey/study/architecture/laba4/benchit/tools/gnuWriter.c /home/andrey/study/architecture/laba4/benchit/bienvhash.c");
      bi_put("BENCHIT_HOSTNAME", "andrey-Easynote-TE69HW");
      bi_put("BENCHIT_IGNORE_PARAMETER_FILE", "0");
      bi_put("BENCHIT_INCLUDES", "-I. -I/home/andrey/study/architecture/laba4/benchit -I/home/andrey/study/architecture/laba4/benchit/tools");
      bi_put("BENCHIT_INTERACTIVE", "0");
      bi_put("BENCHIT_JAVA", "java");
      bi_put("BENCHIT_JAVAC", "javac");
      bi_put("BENCHIT_JAVAC_FLAGS", "");
      bi_put("BENCHIT_JAVAC_FLAGS_HIGH", "-O");
      bi_put("BENCHIT_JAVA_FLAGS", "");
      bi_put("BENCHIT_JAVA_HOME", "/usr/lib/jvm/java-8-openjdk-amd64");
      bi_put("BENCHIT_KERNELBINARY", "/home/andrey/study/architecture/laba4/benchit/bin/tests.0");
      bi_put("BENCHIT_KERNELBINARY_ARGS", " ");
      bi_put("BENCHIT_KERNELNAME", "tests");
      bi_put("BENCHIT_KERNEL_ACCESSES", "1024");
      bi_put("BENCHIT_KERNEL_ALIGNMENT", "256");
      bi_put("BENCHIT_KERNEL_ALLOC", "L");
      bi_put("BENCHIT_KERNEL_COMMENT", " 0B missaligned, alloc: L, hugep.: 0, use: E4, flush: 111 - E2, TLB: 0");
      bi_put("BENCHIT_KERNEL_COUNTERS", "PAPI_L2_DCM");
      bi_put("BENCHIT_KERNEL_CPU_LIST", "0,1,2,3");
      bi_put("BENCHIT_KERNEL_ENABLE_PAPI", "0");
      bi_put("BENCHIT_KERNEL_FLUSH_ACCESSES", "2");
      bi_put("BENCHIT_KERNEL_FLUSH_L1", "1");
      bi_put("BENCHIT_KERNEL_FLUSH_L2", "1");
      bi_put("BENCHIT_KERNEL_FLUSH_L3", "1");
      bi_put("BENCHIT_KERNEL_FLUSH_MODE", "E");
      bi_put("BENCHIT_KERNEL_HUGEPAGES", "0");
      bi_put("BENCHIT_KERNEL_HUGEPAGE_DIR", "/mnt/huge");
      bi_put("BENCHIT_KERNEL_MAX", "200000000");
      bi_put("BENCHIT_KERNEL_MIN", "16000");
      bi_put("BENCHIT_KERNEL_OFFSET", "0");
      bi_put("BENCHIT_KERNEL_RUNS", "4");
      bi_put("BENCHIT_KERNEL_SERIALIZATION", "mfence");
      bi_put("BENCHIT_KERNEL_SHARE_CPU", "7");
      bi_put("BENCHIT_KERNEL_STEPS", "100");
      bi_put("BENCHIT_KERNEL_TIMEOUT", "3600");
      bi_put("BENCHIT_KERNEL_TLB_MODE", "0");
      bi_put("BENCHIT_KERNEL_USE_ACCESSES", "4");
      bi_put("BENCHIT_KERNEL_USE_MODE", "E");
      bi_put("BENCHIT_LD_LIBRARY_PATH", "/home/andrey/study/architecture/laba4/benchit/jbi/jni");
      bi_put("BENCHIT_LIB_ACML", " -lacml");
      bi_put("BENCHIT_LIB_ATLAS", " -latlas");
      bi_put("BENCHIT_LIB_BLAS", "-lblas");
      bi_put("BENCHIT_LIB_ESSL", " -lessl");
      bi_put("BENCHIT_LIB_FFTW3", " -lfftw3");
      bi_put("BENCHIT_LIB_MKL", " -lmkl");
      bi_put("BENCHIT_LIB_MPI", "");
      bi_put("BENCHIT_LIB_PAPI", "");
      bi_put("BENCHIT_LIB_PCL", "");
      bi_put("BENCHIT_LIB_PTHREAD", "");
      bi_put("BENCHIT_LIB_PVM", "");
      bi_put("BENCHIT_LIB_SCSL", " -lscsl");
      bi_put("BENCHIT_LINK", "gcc -o /home/andrey/study/architecture/laba4/benchit/bin/tests.0 *.o -pthread -lm");
      bi_put("BENCHIT_LOCAL_CC", "gcc");
      bi_put("BENCHIT_LOCAL_CC_C_FLAGS", "");
      bi_put("BENCHIT_LOCAL_CC_L_FLAGS", "-lm");
      bi_put("BENCHIT_MANDATORY_FILES", "benchit.c interface.h tools/envhashbuilder.c tools/bienvhash.template.c tools/bienvhash.h tools/stringlib.c tools/stringlib.h tools/bitWriter.c tools/bitWriter.h tools/gnuWriter.c tools/gnuWriter.h tools/output.c tools/output.h ");
      bi_put("BENCHIT_MPICC", "gcc");
      bi_put("BENCHIT_MPICC_C_FLAGS", "");
      bi_put("BENCHIT_MPICC_C_FLAGS_HIGH", "-O3");
      bi_put("BENCHIT_MPICC_C_FLAGS_OMP", "");
      bi_put("BENCHIT_MPICC_C_FLAGS_STD", "-O2");
      bi_put("BENCHIT_MPICC_LD", "gcc");
      bi_put("BENCHIT_MPICC_L_FLAGS", "-lm -lmpi");
      bi_put("BENCHIT_MPICXX_C_FLAGS", "");
      bi_put("BENCHIT_MPICXX_C_FLAGS_HIGH", "-O3");
      bi_put("BENCHIT_MPICXX_C_FLAGS_OMP", "");
      bi_put("BENCHIT_MPICXX_C_FLAGS_STD", "-O2");
      bi_put("BENCHIT_MPICXX_LD", "gcc");
      bi_put("BENCHIT_MPICXX_L_FLAGS", "-lm -lmpi");
      bi_put("BENCHIT_MPIF77", "");
      bi_put("BENCHIT_MPIF77_C_FLAGS", "");
      bi_put("BENCHIT_MPIF77_C_FLAGS_HIGH", "");
      bi_put("BENCHIT_MPIF77_C_FLAGS_OMP", "");
      bi_put("BENCHIT_MPIF77_C_FLAGS_STD", "");
      bi_put("BENCHIT_MPIF77_LD", "");
      bi_put("BENCHIT_MPIF77_L_FLAGS", "");
      bi_put("BENCHIT_MPIRUN", "mpirun");
      bi_put("BENCHIT_MPIXX", "g++");
      bi_put("BENCHIT_NODENAME", "andrey-Easynote-TE69HW");
      bi_put("BENCHIT_NUM_CPUS", "4");
      bi_put("BENCHIT_NUM_PROCESSES", "");
      bi_put("BENCHIT_NUM_THREADS_PER_PROCESS", "");
      bi_put("BENCHIT_OPTIONAL_FILES", "LOCALDEFS/PROTOTYPE_input_architecture LOCALDEFS/PROTOTYPE_input_display ");
      bi_put("BENCHIT_PARAMETER_FILE", "/home/andrey/study/architecture/laba4/benchit/kernel/tests/PARAMETERS");
      bi_put("BENCHIT_PROGRESS_DIR", "progress");
      bi_put("BENCHIT_RUN_ACCURACY", "2");
      bi_put("BENCHIT_RUN_COREDUMPLIMIT", "0");
      bi_put("BENCHIT_RUN_EMAIL_ADDRESS", "");
      bi_put("BENCHIT_RUN_LINEAR", "0");
      bi_put("BENCHIT_RUN_MAX_MEMORY", "3258");
      bi_put("BENCHIT_RUN_OUTPUT_DIR", "/home/andrey/study/architecture/laba4/benchit/output");
      bi_put("BENCHIT_RUN_QUEUENAME", "");
      bi_put("BENCHIT_RUN_REDIRECT_CONSOLE", "");
      bi_put("BENCHIT_RUN_TEST", "0");
      bi_put("BENCHIT_RUN_TIMELIMIT", "3600");
      bi_put("BENCHIT_USE_VAMPIR_TRACE", "0");
      bi_put("BR", "0");
      bi_put("CLUTTER_IM_MODULE", "xim");
      bi_put("CMAKE_PREFIX_PATH", "/opt/ros/kinetic");
      bi_put("COMMENT", "");
      bi_put("COMPILE_GLOBAL", "1");
      bi_put("COMPIZ_BIN_PATH", "/usr/bin/");
      bi_put("COMPIZ_CONFIG_PROFILE", "ubuntu");
      bi_put("CONFIGURE_MODE", "COMPILE");
      bi_put("CURDIR", "/home/andrey/study/architecture/laba4/benchit");
      bi_put("DBUS_SESSION_BUS_ADDRESS", "unix:abstract=/tmp/dbus-RTnNKxMNns");
      bi_put("DEFAULTS_PATH", "/usr/share/gconf/ubuntu.default.path");
      bi_put("DESKTOP_SESSION", "ubuntu");
      bi_put("DEST", "/home/andrey/study/architecture/laba4/benchit/bin/");
      bi_put("DISPLAY", ":0");
      bi_put("GDMSESSION", "ubuntu");
      bi_put("GDM_LANG", "ru_RU");
      bi_put("GNOME_DESKTOP_SESSION_ID", "this-is-deprecated");
      bi_put("GNOME_KEYRING_CONTROL", "");
      bi_put("GNOME_KEYRING_PID", "");
      bi_put("GPG_AGENT_INFO", "/home/andrey/.gnupg/S.gpg-agent:0:1");
      bi_put("GTK2_MODULES", "overlay-scrollbar");
      bi_put("GTK_IM_MODULE", "ibus");
      bi_put("GTK_MODULES", "gail:atk-bridge:unity-gtk-module");
      bi_put("HLL", "");
      bi_put("HOME", "/home/andrey");
      bi_put("IFS", "' 	");
      bi_put("IM_CONFIG_PHASE", "1");
      bi_put("INSTANCE", "");
      bi_put("IS_SCRIPT", "0");
      bi_put("JOB", "unity-settings-daemon");
      bi_put("KERNELBASEDIR", "/home/andrey/study/architecture/laba4/benchit/kernel");
      bi_put("KERNELDIR", "/home/andrey/study/architecture/laba4/benchit/kernel/tests");
      bi_put("KERNELNAME_FULL", "");
      bi_put("KERNEL_COMPILE", "gcc   -O0 -I. -I/home/andrey/study/architecture/laba4/benchit -I/home/andrey/study/architecture/laba4/benchit/tools  -DDEBUGLEVEL=0 -c kernel_main.c");
      bi_put("LANGUAGE", "ru_RU");
      bi_put("LD_LIBRARY_PATH", "/opt/ros/kinetic/lib");
      bi_put("LESSCLOSE", "/usr/bin/lesspipe %s %s");
      bi_put("LESSOPEN", "| /usr/bin/lesspipe %s");
      bi_put("LOCAL_BENCHIT_COMPILER", "gcc");
      bi_put("LOCAL_BENCHIT_COMPILERFLAGS", "  -O0");
      bi_put("LOCAL_FILES_TO_COPY", "");
      bi_put("LOCAL_FILES_TO_DELETE", "");
      bi_put("LOCAL_KERNEL_COMPILER", "gcc");
      bi_put("LOCAL_KERNEL_COMPILERFLAGS", "  -O0");
      bi_put("LOCAL_KERNEL_FILES", "kernel_main.c");
      bi_put("LOCAL_LINKER", "gcc");
      bi_put("LOCAL_LINKERFLAGS", "-pthread -lm");
      bi_put("LOCAL_MODULE_FILES", "arch.c work.c /home/andrey/study/architecture/laba4/benchit/tools/hw_detect/generic.c /home/andrey/study/architecture/laba4/benchit/tools/hw_detect/properties.c /home/andrey/study/architecture/laba4/benchit/tools/hw_detect/x86.c");
      bi_put("LOGNAME", "andrey");
      bi_put("LS_COLORS", "rs=0:di=01;34:ln=01;36:mh=00:pi=40;33:so=01;35:do=01;35:bd=40;33;01:cd=40;33;01:or=40;31;01:mi=00:su=37;41:sg=30;43:ca=30;41:tw=30;42:ow=34;42:st=37;44:ex=01;32:*.tar=01;31:*.tgz=01;31:*.arc=01;31:*.arj=01;31:*.taz=01;31:*.lha=01;31:*.lz4=01;31:*.lzh=01;31:*.lzma=01;31:*.tlz=01;31:*.txz=01;31:*.tzo=01;31:*.t7z=01;31:*.zip=01;31:*.z=01;31:*.Z=01;31:*.dz=01;31:*.gz=01;31:*.lrz=01;31:*.lz=01;31:*.lzo=01;31:*.xz=01;31:*.bz2=01;31:*.bz=01;31:*.tbz=01;31:*.tbz2=01;31:*.tz=01;31:*.deb=01;31:*.rpm=01;31:*.jar=01;31:*.war=01;31:*.ear=01;31:*.sar=01;31:*.rar=01;31:*.alz=01;31:*.ace=01;31:*.zoo=01;31:*.cpio=01;31:*.7z=01;31:*.rz=01;31:*.cab=01;31:*.jpg=01;35:*.jpeg=01;35:*.gif=01;35:*.bmp=01;35:*.pbm=01;35:*.pgm=01;35:*.ppm=01;35:*.tga=01;35:*.xbm=01;35:*.xpm=01;35:*.tif=01;35:*.tiff=01;35:*.png=01;35:*.svg=01;35:*.svgz=01;35:*.mng=01;35:*.pcx=01;35:*.mov=01;35:*.mpg=01;35:*.mpeg=01;35:*.m2v=01;35:*.mkv=01;35:*.webm=01;35:*.ogm=01;35:*.mp4=01;35:*.m4v=01;35:*.mp4v=01;35:*.vob=01;35:*.qt=01;35:*.nuv=01;35:*.wmv=01;35:*.asf=01;35:*.rm=01;35:*.rmvb=01;35:*.flc=01;35:*.avi=01;35:*.fli=01;35:*.flv=01;35:*.gl=01;35:*.dl=01;35:*.xcf=01;35:*.xwd=01;35:*.yuv=01;35:*.cgm=01;35:*.emf=01;35:*.ogv=01;35:*.ogx=01;35:*.aac=00;36:*.au=00;36:*.flac=00;36:*.m4a=00;36:*.mid=00;36:*.midi=00;36:*.mka=00;36:*.mp3=00;36:*.mpc=00;36:*.ogg=00;36:*.ra=00;36:*.wav=00;36:*.oga=00;36:*.opus=00;36:*.spx=00;36:*.xspf=00;36:");
      bi_put("LastChecked", "");
      bi_put("MANDATORY_PATH", "/usr/share/gconf/ubuntu.mandatory.path");
      bi_put("MODULE_COMPILE", "gcc   -O0 -I. -I/home/andrey/study/architecture/laba4/benchit -I/home/andrey/study/architecture/laba4/benchit/tools  -DDEBUGLEVEL=0 -c arch.c work.c /home/andrey/study/architecture/laba4/benchit/tools/hw_detect/generic.c /home/andrey/study/architecture/laba4/benchit/tools/hw_detect/properties.c /home/andrey/study/architecture/laba4/benchit/tools/hw_detect/x86.c");
      bi_put("OLDCWD", "/home/andrey/study/architecture/laba4/benchit");
      bi_put("OLDIR", "/home/andrey/study/architecture/laba4/benchit");
      bi_put("OLDPWD", "/home/andrey/study/architecture/laba4/benchit/kernel/tests");
      bi_put("OMP_DYNAMIC", "FALSE");
      bi_put("OMP_NESTED", "FALSE");
      bi_put("OMP_NUM_THREADS", "1");
      bi_put("OPTIND", "1");
      bi_put("PATH", "/home/andrey/study/architecture/laba4/benchit/tools:/opt/ros/kinetic/bin:/home/andrey/bin:/home/andrey/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin");
      bi_put("PKG_CONFIG_PATH", "/opt/ros/kinetic/lib/pkgconfig");
      bi_put("PPID", "25870");
      bi_put("PS1", "$ ");
      bi_put("PS2", "> ");
      bi_put("PS4", "+ ");
      bi_put("PWD", "/home/andrey/study/architecture/laba4/benchit/tools");
      bi_put("PYTHONPATH", "/opt/ros/kinetic/lib/python2.7/dist-packages");
      bi_put("QT4_IM_MODULE", "xim");
      bi_put("QT_ACCESSIBILITY", "1");
      bi_put("QT_IM_MODULE", "ibus");
      bi_put("QT_LINUX_ACCESSIBILITY_ALWAYS_ON", "1");
      bi_put("QT_QPA_PLATFORMTHEME", "appmenu-qt5");
      bi_put("ROSLISP_PACKAGE_DIRECTORIES", "");
      bi_put("ROS_DISTRO", "kinetic");
      bi_put("ROS_ETC_DIR", "/opt/ros/kinetic/etc/ros");
      bi_put("ROS_MASTER_URI", "http://localhost:11311");
      bi_put("ROS_PACKAGE_PATH", "/opt/ros/kinetic/share");
      bi_put("ROS_ROOT", "/opt/ros/kinetic/share/ros");
      bi_put("SCRIPTNAME", "COMPILE.SH");
      bi_put("SESSION", "ubuntu");
      bi_put("SESSIONTYPE", "gnome-session");
      bi_put("SESSION_MANAGER", "local/andrey-Easynote-TE69HW:@/tmp/.ICE-unix/1507,unix/andrey-Easynote-TE69HW:/tmp/.ICE-unix/1507");
      bi_put("SHELL", "/bin/bash");
      bi_put("SHELLSCRIPT_DEBUG", "0");
      bi_put("SHLVL", "1");
      bi_put("SSH_AUTH_SOCK", "/run/user/1000/keyring/ssh");
      bi_put("TERM", "xterm-256color");
      bi_put("UPSTART_EVENTS", "xsession started");
      bi_put("UPSTART_INSTANCE", "");
      bi_put("UPSTART_JOB", "unity7");
      bi_put("UPSTART_SESSION", "unix:abstract=/com/ubuntu/upstart-session/1000/1032");
      bi_put("USEJAVA", "0");
      bi_put("USER", "andrey");
      bi_put("VTE_VERSION", "4205");
      bi_put("WINDOWID", "85983242");
      bi_put("XAUTHORITY", "/home/andrey/.Xauthority");
      bi_put("XDG_CONFIG_DIRS", "/etc/xdg/xdg-ubuntu:/usr/share/upstart/xdg:/etc/xdg");
      bi_put("XDG_CURRENT_DESKTOP", "Unity");
      bi_put("XDG_DATA_DIRS", "/usr/share/ubuntu:/usr/share/gnome:/usr/local/share/:/usr/share/:/var/lib/snapd/desktop");
      bi_put("XDG_GREETER_DATA_DIR", "/var/lib/lightdm-data/andrey");
      bi_put("XDG_MENU_PREFIX", "gnome-");
      bi_put("XDG_RUNTIME_DIR", "/run/user/1000");
      bi_put("XDG_SEAT", "seat0");
      bi_put("XDG_SEAT_PATH", "/org/freedesktop/DisplayManager/Seat0");
      bi_put("XDG_SESSION_DESKTOP", "ubuntu");
      bi_put("XDG_SESSION_ID", "c1");
      bi_put("XDG_SESSION_PATH", "/org/freedesktop/DisplayManager/Session0");
      bi_put("XDG_SESSION_TYPE", "x11");
      bi_put("XDG_VTNR", "7");
      bi_put("XMODIFIERS", "@im=ibus");
      bi_put("_", "./benchit/COMPILE.SH");
      bi_put("_CMDLINE_VARLIST", "BENCHIT_KERNELBINARY BENCHIT_KERNELBINARY_ARGS BENCHIT_CMDLINE_ARG_FILENAME_COMMENT BENCHIT_CMDLINE_ARG_PARAMETER_FILE BENCHIT_CMDLINE_ARG_IGNORE_PARAMETER_FILE BENCHIT_NODENAME BENCHIT_CROSSCOMPILE BENCHIT_CMDLINE_ARG_NUM_CPUS BENCHIT_CMDLINE_ARG_NUM_PROCESSES BENCHIT_CMDLINE_ARG_NUM_THREADS_PER_PROCESS BENCHIT_CMDLINE_ARG_RUN_CLEAN BENCHIT_CMDLINE_ARG_RUN_COREDUMPLIMIT BENCHIT_CMDLINE_ARG_RUN_EMAIL_ADDRESS BENCHIT_CMDLINE_ARG_RUN_MAX_MEMORY BENCHIT_CMDLINE_ARG_RUN_QUEUENAME BENCHIT_CMDLINE_ARG_RUN_QUEUETIMELIMIT BENCHIT_CMDLINE_ARG_RUN_REDIRECT_CONSOLE BENCHIT_CMDLINE_ARG_RUN_TEST BENCHIT_CMDLINE_ARG_RUN_USE_MPI BENCHIT_CMDLINE_ARG_RUN_USE_OPENMP ");
      bi_put("_VARLIST", "'BENCHITROOT");
      bi_put("cFlags", "");
      bi_put("cFlagsOpt", "-O0");
      bi_put("file", "/home/andrey/study/architecture/laba4/benchit/tools/hw_detect/x86.c");
      bi_put("flags", "");
      bi_put("lFlags", "-lm");
      bi_put("myfile", "tools/output.h");
      bi_put("myval", "");
      bi_put("myvar", "BENCHIT_CMDLINE_ARG_RUN_USE_OPENMP");
      bi_put("retval", "0");
      bi_put("tmpCompiler", "gcc");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_CMDLINEPARAMS", "NO REVISION (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_ERROR_H", "NO REVISION (Wed Aug 25 17:53:18 2010)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_ALIGNED_MEMORY_H", "NO REVISION (Wed Aug 25 17:53:40 2010)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_BIENVHASH_TEMPLATE_C", "NO REVISION (Wed Jul 10 23:38:47 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_COMPILERVERSION", "NO REVISION, UNABLE TO READ (Sun Jan 19 06:00:39 2014)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_GNUWRITER_C", "NO REVISION (Mon Aug 12 23:58:00 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_IRODS", "NO REVISION, UNABLE TO READ (Sun Jan 19 06:00:39 2014)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_BIENVHASH_H", "NO REVISION (Wed Jul 10 23:38:47 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_REFERENCE_RUN_SH", "NO REVISION (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_BITWRITER_H", "NO REVISION (Tue Jul  2 22:32:22 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_REPEAT_H", "NO REVISION (Mon Sep  2 20:18:08 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_BMERGE_SH", "NO REVISION (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_TMP_ENV", "NO REVISION (Wed Nov 22 17:19:10 2017)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_ENVHASHBUILDER", "NO REVISION (Wed Nov 22 17:19:10 2017)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_HELPER_SH", "1.14 (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_LOC_CONVERT_SH", "NO REVISION (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_BIENVHASH_TEMPLATE_JAVA", "NO REVISION (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_STRINGLIB_H", "NO REVISION (Wed Aug 25 17:53:18 2010)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_FEATURES", "NO REVISION (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_QUICKVIEW_SH", "NO REVISION (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_STRINGLIB_C", "NO REVISION (Thu Aug  8 19:25:10 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_CHANGE_SH_SH", "NO REVISION (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_FIRSTTIME", "NO REVISION (Wed Aug 25 17:53:18 2010)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_FILEVERSION", "NO REVISION (Wed Nov 22 17:19:10 2017)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_OUTPUT_H", "NO REVISION (Wed Jul 10 23:38:47 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_LOC_REPL", "NO REVISION (Thu Nov 12 19:13:49 2009)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_ENVIRONMENTS", "NO REVISION, UNABLE TO READ (Sun Jan 19 06:00:39 2014)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_BENCHSCRIPT_H", "NO REVISION (Wed Aug 25 17:53:40 2010)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_HW_DETECT", "NO REVISION, UNABLE TO READ (Wed Nov 15 17:51:54 2017)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_FILEVERSION_C", "*/ (Mon Jun 17 19:57:27 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_GNUWRITER_H", "NO REVISION (Wed Jul 10 23:38:47 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_CBLAS", "NO REVISION, UNABLE TO READ (Sun Jan 19 06:00:36 2014)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_OUTPUT_C", "NO REVISION (Tue Jul  2 23:05:03 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_BITWRITER_C", "NO REVISION (Mon Aug 12 23:58:00 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_BENCHSCRIPT_C", "NO REVISION (Mon Jun 17 19:57:27 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_BLAS", "NO REVISION, UNABLE TO READ (Sun Jan 19 06:00:36 2014)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_CONFIGURE", "NO REVISION (Tue Jul 16 22:31:45 2013)");
      bi_put("BENCHIT_KERNEL_FILE_VERSION_ENVHASHBUILDER_C", "NO REVISION (Mon Jun 17 19:57:27 2013)");
   }
}
