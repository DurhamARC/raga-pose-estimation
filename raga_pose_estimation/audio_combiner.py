
import os
from pymediainfo import MediaInfo

class Audio:
    def __init__(self, input_video_path, output_directory):
        """Initializes a Visualizer instance with a set of parts to display and
        an output directory.

        Parameters
        ----------
        output_directory : str
            Path to video output folder. Directory will be created if it
            doesn't exist.

        Returns
        -------
        Visualizer instance

        """
        self.output_directory = output_directory
        self.input_video_path = input_video_path
    
    def _has_audio(self, input_video_path):
        fileInfo = MediaInfo.parse(input_video_path)
        return any([track.track_type == 'Audio' for track in fileInfo.tracks])

    def _extract_audio(self, input_video_path, output_directory):
        """
        Parameters
        ---------
        Return:
        None
        """
        command = "ffmpeg -i "
        command += input_video_path
        command += " -vn -acodec copy "
        command += output_directory
        command += "/audio.aac"
        print(command)
        os.system(command)
        return None

    def _attach_audio(self, output_directory):
        command = "ffmpeg -i "
        command += output_directory
        command += "/video_overlay.mp4"
        command += " -i "
        command += output_directory 
        command += "/audio.aac"
        command += " -c:v copy -c:a aac "
        command += output_directory 
        command += "/video_overlay_with_sound.mp4"
        os.system(command)
        return None

    def audio_combiner(self, input_video_path, output_directory):
        if self._has_audio(input_video_path):
            self._extract_audio(input_video_path, output_directory)
            self._attach_audio(output_directory)
        else:
            print("No audio found in the input video.")