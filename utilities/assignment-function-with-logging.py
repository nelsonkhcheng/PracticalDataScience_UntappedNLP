from datetime import datetime
import os 
from pathlib import Path

# Make sure ALL functionality is in this function, no separate functions
def process_teams (input_files):
  '''The main data array. '''
  data_array = []

  ''' TODO by you '''

  # print out running information to file so I can track what the function is doing, but I want to easily disable this
  # on submission
  enableDebugFile = True
  debugFileName = "debug/challenge1." + datetime.now().strftime("%Y%m%d.%H%M%S") + ".log"
  functionName = "process_teams()"
  debugLogs = list()
  print(debugFileName)
  
  # Note: scope allows inner functions to access outer
  def log_message(message):
    if enableDebugFile:
      debugLogs.append("Message: " + message)

  def log_error(message):
    if enableDebugFile:
      debugLogs.append("Error: " + message)

  def write_debug_logs():
    # Create the debug folder, if it doesn't already exist
    if not os.path.exists("debug/"):
      Path("debug/").mkdir(parents=True,exist_ok=True)
    # write the log messages to a new debug file
    with open(debugFileName,'a') as out_file:
      for log in debugLogs:        
        out_file.write(log + '\n')
      out_file.write('\n\n')



  try:
    print(functionName + " started")
    log_message(functionName + " started.")
    
    ''' START Actual processing starts here '''

    
    ''' END Actual processing starts here '''
  except:
    log_error("Exception occurred in " + functionName + ".")
    log_error(traceback.format_exc())
  finally:
    print(functionName + " function ended.")
    log_message(functionName + " function ended.")
    
    # At the end, write out all the log messages to file
    write_debug_logs()    

  return data_array