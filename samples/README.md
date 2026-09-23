# `samples/` - drop-in media for local testing without CCTV hardware

## `videos/` - mock camera sources

Put a video file here (e.g. `lecture.mp4`) containing footage of people
walking past the camera. The frame reader treats a file source as an endless
stream: on EOF it loops back to frame 0, so a 30-second clip simulates a live
camera indefinitely.

```bash
# register it as a mock entry gate
docker compose exec api python scripts/seed.py \
  --camera --name "Main Gate" --type entry \
  --video /samples/videos/lecture.mp4 --location "Main entrance"

# start processing
curl -X POST http://localhost:8000/cameras/1/start \
  -H "Authorization: Bearer $TOKEN"
```

Any common container format works (mp4/mkv/avi) - decoding goes through
FFmpeg. For RTSP CCTV use `--rtsp rtsp://user:pass@camera-ip/stream1` instead.

## `faces/<registration_no>/*.jpg` - enrollment photos

```
faces/
├── 21CSE001/          <- registration number = folder name
│   ├── front.jpg       <- 3-5 clear, front-facing photos
│   ├── left.jpg
│   └── right.jpg
└── 21CSE002/
    └── ...
```

```bash
# enroll every subfolder
docker compose exec api python scripts/seed.py --enroll-dir /samples/faces
```

Good photos: face clearly visible, decent lighting, one person per photo,
mix of angles/lighting for robustness. The GPU service detects the largest
face per photo and stores one 512-d embedding per photo (recognition uses
best-match across all of them).
