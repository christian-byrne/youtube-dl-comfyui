import os
import platform
import mimetypes

import torchaudio
import torch
from yt_dlp import YoutubeDL
import yt_dlp.utils

from .parse_custom_cli_args import cli_to_api
import folder_paths

from typing import Optional, List


class YoutubeDLNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "youtube_link": (
                    "STRING",
                    {"default": "https://www.youtube.com/watch?v=6bALJxjL8jw"},
                ),
                "playlist_start": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "playlist_end": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
            },
            "optional": {
                "audio_quality": (
                    "FLOAT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "display": "slider",
                    },
                ),
                "delete_after": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "toggle_on": "Delete file after extraction",
                        "toggle_off": "Save file permanently",
                    },
                ),
                "random_from_playlist": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "toggle_on": "Shuffle Playlist",
                        "toggle_off": "Do not shuffle Playlist",
                    },
                ),
                "yt_dlp_cli_args": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "audio"

    def main(
        self,
        youtube_link,
        playlist_start,
        playlist_end,
        audio_quality: Optional[float] = None,
        delete_after: bool = False,
        random_from_playlist: bool = False,
        yt_dlp_cli_args: Optional[str] = None,
    ):

        self.is_windows = platform.system() == "Windows"
        input_dir = folder_paths.get_input_directory()
        ydl_opts = {
            "format": "best",
            "playliststart": playlist_start,
            "playlistend": playlist_end,
            "outtmpl": f"{input_dir}/%(title)s.%(ext)s",
            # "writesubtitles": True,
            # "writeautomaticsub": True,
            # "embedsubtitles": False,
        }

        if audio_quality:
            ydl_opts["audioquality"] = int(round(audio_quality))
        if random_from_playlist:
            ydl_opts["playlist_random"] = True
        if self.is_windows:
            ydl_opts["windowsfilenames"] = True
        if yt_dlp_cli_args:
            ydl_opts.update(cli_to_api(yt_dlp_cli_args))

        output = []
        output_paths = []
        batch_sample_rate = None

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_link, download=True)
            sanitized = ydl.sanitize_info(info)

            if self.is_playlist(sanitized):
                for path in self.get_playlist_entry_titles(sanitized):
                    abs_path = self.resolve_path(path)
                    output_paths.append(abs_path)
                    waveform, sample_rate = self.path_to_waveform(abs_path)
                    if batch_sample_rate is None:
                        batch_sample_rate = sample_rate
                    elif batch_sample_rate != sample_rate:
                        raise ValueError(
                            f"Sample rate mismatch in playlist items: {batch_sample_rate} != {sample_rate}"
                        )
                    output.append(waveform)
            else:
                abs_path = self.resolve_path(sanitized["title"])
                output_paths.append(abs_path)
                waveform, sample_rate = self.path_to_waveform(abs_path)
                batch_sample_rate = sample_rate
                output.append(waveform)

        if len(output) > 1:
            batched_waveform = self.pad_cat(output)
        else:
            batched_waveform = output[0]

        audio = {"waveform": batched_waveform, "sample_rate": batch_sample_rate}

        if delete_after:
            self.delete_files(output_paths)

        return (audio,)

    def is_playlist(self, res: dict) -> bool:
        has_playlist_basename = (
            "webpage_url_basename" in res and res["webpage_url_basename"] == "playlist"
        )
        has_playlist_count = "playlist_count" in res and res["playlist_count"] > 0
        return has_playlist_basename or has_playlist_count

    def pad_cat(self, waveforms: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            waveforms: List[torch.Tensor]

        Returns:
            padded_waveform: torch.Tensor [B, C, F]
        """
        max_len = max(waveform.shape[-1] for waveform in waveforms)
        padded_waveforms = []
        for waveform in waveforms:
            padded_waveform = torch.nn.functional.pad(
                waveform, (0, max_len - waveform.shape[-1])
            )
            padded_waveforms.append(padded_waveform)
        return torch.cat(padded_waveforms, dim=0)

    def delete_files(self, paths):
        for path in paths:
            os.remove(path)

    def match_file(self, basename: str, mime_type: str = "video") -> str:
        input_dir = folder_paths.get_input_directory()
        windows_basename = yt_dlp.utils.sanitize_filename(basename)
        for fi in os.listdir(input_dir):
            root, _ = os.path.splitext(fi)
            matched_windows = self.is_windows and root == windows_basename
            matched = root == basename
            if matched or matched_windows:
                mtype = mimetypes.guess_type(os.path.join(input_dir, fi))[0]
                if mtype and mime_type in mtype:
                    return fi

        raise FileNotFoundError(
            f"Could not find file in input_dir {input_dir} with basename: {basename}"
        )

    def resolve_path(self, basename: str) -> str:
        input_dir = folder_paths.get_input_directory()
        filename = self.match_file(basename)
        return os.path.join(input_dir, filename)

    def path_to_waveform(self, path: str) -> torch.Tensor:
        """
        Returns:
            waveform: torch.Tensor [B, C, F]
            sample_rate: int

        """
        waveform, sample_rate = torchaudio.load(path)
        return waveform.unsqueeze(0), sample_rate

    def get_playlist_entry_titles(self, res: dict) -> List[str]:
        if "entries" not in res:
            raise KeyError(
                f"Expected this to be a playlist but there are is no 'entries' key in the response: {res}"
            )

        entries = res["entries"]
        paths = []
        for entry in entries:
            paths.append(entry["title"])
        return paths


NODE_CLASS_MAPPINGS = {"YoutubeDLNode": YoutubeDLNode}
