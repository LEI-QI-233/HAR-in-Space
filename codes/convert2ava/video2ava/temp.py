import os
import ffmpeg
import shutil

def extract_all_frames_with_structure(input_root, output_root, fps=30):
    # 先获取所有 .mp4 文件的路径
    video_files = []
    for dirpath, _, filenames in os.walk(input_root):
        for filename in filenames:
            if filename.endswith('.mp4'):
                video_files.append(os.path.join(dirpath, filename))
    
    total_videos = len(video_files)
    processed_videos = 0

    for input_path in video_files:
        processed_videos += 1

        # 相对于输入根目录的路径
        rel_path = os.path.relpath(input_path, input_root)
        rel_dir = os.path.dirname(rel_path)
        clip_name = os.path.splitext(os.path.basename(input_path))[0]

        # 输出路径结构：output/movie/Apollo13/000003/
        out_dir = os.path.join(output_root, rel_dir, clip_name)
        os.makedirs(out_dir, exist_ok=True)

        out_pattern = os.path.join(out_dir, f'{clip_name}_%06d.jpg')

        print(f"[{processed_videos}/{total_videos}] processing {clip_name}...")

        try:
            (
                ffmpeg
                .input(input_path)
                .output(out_pattern, r=fps, qscale=1)
                .run(overwrite_output=True, quiet=True)
            )
            print(f"[OK] Done: {clip_name}")
        except ffmpeg.Error as e:
            print(f"[ERROR] Failed on {clip_name}")
            print(e.stderr.decode())

def clear_output_folder(output_root):
    if os.path.exists(output_root):
        print(f"[WARN] Clearing existing output folder: {output_root}")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)

def run():
    input = "/home/lqi/lqi_temp/trainingspace/videoprocess/testinput"
    output = "/home/lqi/lqi_temp/trainingspace/videoprocess/output"
    clear_output_folder(output)
    extract_all_frames_with_structure(input, output)

if __name__ == "__main__":
    run()
