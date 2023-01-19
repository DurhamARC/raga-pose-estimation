
import os
from pymediainfo import MediaInfo

class Audio:
    def __init__(self, input_video_path, output_directory):
        self.output_directory = output_directory
        self.input_video_path = input_video_path
    
    def _has_audio(self, input_video_path):
        """Check if video has audio.

        Parameters
        ----------
        input_video_path : str
            String signifying the path to the input video

        Returns
        -------
        Booleen value for if audio found

        """
        fileInfo = MediaInfo.parse(input_video_path)
        return any([track.track_type == 'Audio' for track in fileInfo.tracks])

    def _extract_audio(self, input_video_path, output_directory):
        """Extracts audio from an input video

        Parameters
        ----------
        input_video_path : Str
            String signifying the path to the input video

        output_directory: Str
            String signifying output directory path

        Returns
        -------
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
        """Attaches audio to the overlay video in the output directory

        Parameters
        ----------
        input_video_path : Str
            String signifying the path to the input video

        output_directory: Str
            String signifying output directory path

        Returns
        -------
        None

        """
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
        """Combines audio to output overlay video

        Parameters
        ----------
        input_video_path : Str
            String signifying the path to the input video

        output_directory: Str
            String signifying output directory path

        Returns
        -------
        None

        """
        if self._has_audio(input_video_path):
            self._extract_audio(input_video_path, output_directory)
            self._attach_audio(output_directory)
        else:
            print("No audio found in the input video.")