# Instructions to run chunk-based speaker diarization

1. Use the function `diarization_init` in file diarization.py to create an diarization instance. The input of the function is a list of enrollment speakers (paths of wav files). The function will return an diarization instance.

2. Use the function `diarization_predict` in file diarization.py to run chunk-based speaker diarization. The inputs of the function includes:

`sd`: The instance returned by function `diarization_init`.  
`input_file`: the path of input file (1-channel wav file whose sampling rate is 16k).  
`output_file`: the path of output file which will be a JSON file.  
