#!/usr/bin/env python3
"""
Interactive Audio Processor - Python Version
A comprehensive audio processing tool using Cleanvoice API
"""
# NEW imports for merge
import re
from typing import List
import pathlib

# Cloudinary (lazy import inside method if not installed)

import os
import sys
import glob
import requests
from pathlib import Path
from typing import Optional, Tuple
from audio_processor import AudioProcessor


class InteractiveAudioProcessor:
    """Interactive audio processor that can be used from CLI or imported into an API."""

    def __init__(self, api_key: str):
        self.processor = AudioProcessor(api_key)
        self._override_input_path: Optional[str] = None  # optional: path to uploaded file

        self.function_map = {
            'rm bg': {
                'func': 'denoise_audio',
                'name': 'Remove Background Noise',
                'params': [False, {'normalize': True}]
            },
            'rm silence': {
                'func': 'remove_silences',
                'name': 'Remove Long Silences',
                'params': [{'export_format': 'auto'}]
            },
            'rm stutter': {
                'func': 'remove_stutters',
                'name': 'Remove Stutters',
                'params': [{'export_format': 'auto'}]
            },
            'rm filler': {
                'func': 'remove_fillers',
                'name': 'Remove Filler Words',
                'params': [{'export_format': 'auto'}]
            },
            'rm mouth': {
                'func': 'remove_mouth_sounds',
                'name': 'Remove Mouth Sounds',
                'params': [{'export_format': 'auto'}]
            },
            'rm hesitation': {
                'func': 'remove_hesitations',
                'name': 'Remove Hesitations',
                'params': [{'export_format': 'auto'}]
            },
            'rm breath': {
                'func': 'reduce_breath_sounds',
                'name': 'Reduce Breath Sounds',
                'params': [-80, {'export_format': 'auto'}]
            },
            'normalize': {
                'func': 'normalize_audio',
                'name': 'Normalize Audio Levels',
                'params': [-16, {'export_format': 'auto'}]
            },
            'ai enhance': {
                'func': 'enhance_with_ai',
                'name': 'AI Sound Enhancement',
                'params': [{'export_format': 'auto'}]
            },
            'preserve music': {
                'func': 'preserve_music',
                'name': 'Preserve Music Segments',
                'params': [{'export_format': 'auto'}]
            },
            'transcribe': {
                'func': 'transcribe_audio',
                'name': 'Transcribe Audio to Text',
                'params': [{'export_format': 'auto'}]
            },
            'comprehensive': {
                'func': 'enhance_audio_comprehensive',
                'name': 'Comprehensive Enhancement',
                'params': [{'export_format': 'auto', 'transcription': True}]
            }
        }

    # ----------------- File handling -----------------
    def set_input_file(self, path: Optional[str]) -> None:
        """Override the input file path for the next processing call."""
        self._override_input_path = path

    def find_audio_file(self) -> Optional[Tuple[str, str]]:
        """
        Find the audio file to process.
        Priority:
          1. If set_input_file() was called, use that file.
          2. Otherwise, look for a default sample.m4a in the current directory.
        """
        if self._override_input_path and os.path.exists(self._override_input_path):
            return self._override_input_path, os.path.basename(self._override_input_path)

        target_file = "sample.m4a"
        if os.path.exists(target_file):
            return target_file, os.path.basename(target_file)

        return None

    def download_file(self, url: str, output_path: str) -> bool:
        """Download a file from URL to local path."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return True
        except Exception as e:
            print(f'❌ Error downloading file: {e}')
            return False

    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    # ----------------- Core processing -----------------
    async def process_audio_file(self, command: str):
        """Process audio file with the given command."""
        try:
            config = self.function_map.get(command)
            if not config:
                print('❌ Unknown command')
                return

            # Get audio file
            file_info = self.find_audio_file()
            if not file_info:
                print('❌ No audio file found')
                return
            file_path, filename = file_info

            # Upload
            signed_url = self.processor.upload_file(file_path, filename)

            # Process
            params = config['params'].copy()
            method = getattr(self.processor, config['func'])
            processing_job = method(signed_url, *params)

            # Wait
            result = self.processor.wait_for_completion(processing_job['id'], 5000, 300000)

            # Download
            download_url = result['result']['download_url']
            input_ext = Path(filename).suffix.lower()
            output_ext = input_ext
            output_path = f"{Path(filename).stem}-{command.replace(' ', '-')}{output_ext}"

            if self.download_file(download_url, output_path):
                size = os.path.getsize(output_path)
                print(f"Saved: {output_path} ({self.format_file_size(size)})")
                return output_path

        except Exception as e:
            print(f'❌ Error processing audio: {e}')
            return None