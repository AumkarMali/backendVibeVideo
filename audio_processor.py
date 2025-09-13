import requests
import os
import time
from typing import Union, List, Dict, Any, Optional


class AudioProcessor:
    """
    AudioProcessor - A comprehensive audio processing class using Cleanvoice API
    Provides all the functions for audio enhancement and processing
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://api.cleanvoice.ai/v2'
        self.headers = {
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        }
    
    def process_audio(self, files: Union[str, List[str]], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process audio files with customizable editing options
        
        Args:
            files: Single file URL or list of file URLs
            config: Configuration options for processing
            
        Returns:
            Edit job response with task ID
        """
        try:
            if config is None:
                config = {}
                
            file_array = files if isinstance(files, list) else [files]
            
            payload = {
                'input': {
                    'files': file_array,
                    'config': config
                }
            }
            
            response = requests.post(
                f'{self.base_url}/edits',
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = e.response.json().get('message', str(e)) if e.response else str(e)
            raise Exception(f'Failed to process audio: {error_msg}')
    
    def remove_silences(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Detect and remove long silences in audio"""
        if additional_config is None:
            additional_config = {}
        config = {
            'long_silences': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def remove_stutters(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Identify and remove stutters from speech"""
        if additional_config is None:
            additional_config = {}
        config = {
            'stutters': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def remove_fillers(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Remove filler words like 'um' or 'uh'"""
        if additional_config is None:
            additional_config = {}
        config = {
            'fillers': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def remove_mouth_sounds(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Eliminate unwanted mouth sounds"""
        if additional_config is None:
            additional_config = {}
        config = {
            'mouth_sounds': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def remove_hesitations(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Remove hesitation noises in speech"""
        if additional_config is None:
            additional_config = {}
        config = {
            'hesitations': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def mute_segments(self, files: Union[str, List[str]], mute_lufs: int = -80, additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mute selected segments instead of cutting them"""
        if additional_config is None:
            additional_config = {}
        config = {
            'muted': True,
            'mute_lufs': mute_lufs,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def denoise_audio(self, files: Union[str, List[str]], keep_music: bool = False, additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Remove background noise while preserving clarity"""
        if additional_config is None:
            additional_config = {}
        config = {
            'remove_noise': True,
            'keep_music': keep_music,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def preserve_music(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Avoid removing music segments during edits"""
        if additional_config is None:
            additional_config = {}
        config = {
            'keep_music': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def reduce_breath_sounds(self, files: Union[str, List[str]], mute_lufs: int = -80, additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Lower the volume of breath sounds naturally"""
        if additional_config is None:
            additional_config = {}
        config = {
            'breath': True,
            'mute_lufs': mute_lufs,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def normalize_audio(self, files: Union[str, List[str]], target_lufs: int = -16, additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adjust audio levels to a target loudness"""
        if additional_config is None:
            additional_config = {}
        config = {
            'normalize': True,
            'target_lufs': target_lufs,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def apply_autoeq(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply automatic EQ adjustments (legacy option)"""
        if additional_config is None:
            additional_config = {}
        config = {
            'autoeq': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def enhance_with_ai(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use AI-based sound enhancement for better quality"""
        if additional_config is None:
            additional_config = {}
        config = {
            'sound_studio': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def set_mute_lufs(self, files: Union[str, List[str]], mute_lufs: int, additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Define loudness threshold for muting audio segments"""
        if mute_lufs > 0:
            raise ValueError('mute_lufs must be a negative integer')
        
        if additional_config is None:
            additional_config = {}
        config = {
            'mute_lufs': mute_lufs,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def set_target_lufs(self, files: Union[str, List[str]], target_lufs: int, additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Set target loudness for normalization"""
        if target_lufs >= 0:
            raise ValueError('target_lufs must be less than 0')
        
        if additional_config is None:
            additional_config = {}
        config = {
            'target_lufs': target_lufs,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def export_audio(self, files: Union[str, List[str]], format: str = "auto", additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Export processed audio in desired format"""
        valid_formats = ["auto", "mp3", "wav", "flac", "m4a"]
        if format not in valid_formats:
            raise ValueError(f'Invalid export format. Must be one of: {", ".join(valid_formats)}')
        
        if additional_config is None:
            additional_config = {}
        config = {
            'export_format': format,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def transcribe_audio(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Convert speech in audio to text"""
        if additional_config is None:
            additional_config = {}
        config = {
            'transcription': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def summarize_audio(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a summary of transcribed audio content"""
        if additional_config is None:
            additional_config = {}
        config = {
            'transcription': True,
            'summarize': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def create_social_content(self, files: Union[str, List[str]], additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare audio clips for social media sharing"""
        if additional_config is None:
            additional_config = {}
        config = {
            'transcription': True,
            'summarize': True,
            'social_content': True,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def merge_tracks(self, files: List[str], normalize: bool = True, additional_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Merge multiple audio tracks into a single file"""
        if not isinstance(files, list) or len(files) < 2:
            raise ValueError('merge_tracks requires a list of at least 2 file URLs')
        
        if additional_config is None:
            additional_config = {}
        config = {
            'merge': True,
            'normalize': normalize,
            **additional_config
        }
        return self.process_audio(files, config)
    
    def upload_file(self, file_path: str, filename: str) -> str:
        """
        Upload a file directly to Cleanvoice servers
        
        Args:
            file_path: Local file path
            filename: Desired filename
            
        Returns:
            Signed URL for the uploaded file
        """
        try:
            # Step 1: Get signed URL
            upload_response = requests.post(
                f'{self.base_url}/upload?filename={filename}',
                headers={'X-API-Key': self.api_key}
            )
            upload_response.raise_for_status()
            
            signed_url = upload_response.json()['signedUrl']
            
            # Step 2: Upload file to signed URL
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            upload_put_response = requests.put(
                signed_url,
                data=file_data,
                headers={'Content-Type': 'application/octet-stream'}
            )
            upload_put_response.raise_for_status()
            
            return signed_url
        except requests.exceptions.RequestException as e:
            error_msg = e.response.json().get('message', str(e)) if e.response else str(e)
            raise Exception(f'Failed to upload file: {error_msg}')
    
    def get_edit_status(self, edit_id: str) -> Dict[str, Any]:
        """Check the status of an edit job"""
        try:
            response = requests.get(
                f'{self.base_url}/edits/{edit_id}',
                headers={'X-API-Key': self.api_key}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = e.response.json().get('message', str(e)) if e.response else str(e)
            raise Exception(f'Failed to get edit status: {error_msg}')
    
    def wait_for_completion(self, edit_id: str, poll_interval: int = 5000, max_wait_time: int = 300000) -> Dict[str, Any]:
        """
        Wait for edit completion and return results
        
        Args:
            edit_id: Edit job ID
            poll_interval: Polling interval in milliseconds
            max_wait_time: Maximum wait time in milliseconds (default: 5 minutes)
            
        Returns:
            Final edit results
        """
        start_time = time.time() * 1000  # Convert to milliseconds
        
        while (time.time() * 1000) - start_time < max_wait_time:
            status = self.get_edit_status(edit_id)
            
            if status['status'] == 'SUCCESS':
                return status
            elif status['status'] == 'FAILURE':
                raise Exception('Edit job failed')
            
            # Wait before next poll
            time.sleep(poll_interval / 1000)  # Convert to seconds
        
        raise Exception('Edit job timed out')
    
    def delete_edit(self, edit_id: str) -> Dict[str, Any]:
        """Delete an edit job and its associated files"""
        try:
            response = requests.delete(
                f'{self.base_url}/edits/{edit_id}',
                headers={'X-API-Key': self.api_key}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = e.response.json().get('message', str(e)) if e.response else str(e)
            raise Exception(f'Failed to delete edit: {error_msg}')
    
    def enhance_audio_comprehensive(self, files: Union[str, List[str]], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive audio enhancement with all recommended settings
        
        Args:
            files: File URL(s)
            options: Enhancement options to override defaults
            
        Returns:
            Edit job response
        """
        if options is None:
            options = {}
            
        default_config = {
            # Core audio cleaning
            'long_silences': True,
            'stutters': True,
            'fillers': True,
            'mouth_sounds': True,
            'hesitations': True,
            'remove_noise': True,
            'breath': True,
            
            # Audio enhancement
            'normalize': True,
            'target_lufs': -16,
            'sound_studio': True,
            
            # Export settings
            'export_format': "mp3",
            
            # Optional features
            'transcription': False,
            'summarize': False,
            'social_content': False,
            'export_timestamps': False,
            
            **options
        }
        
        return self.process_audio(files, default_config)
