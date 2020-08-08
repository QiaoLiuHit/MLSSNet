function paths = env_paths_tracking(varargin)
    paths.net_base = './pretrained/'; % change it  to yours
    paths.eval_set_base = '/media/joe/000FA49A000EC49F/PTB-TIR-Evaluation_toolkit/trackers/MLSSNet/data/'; % change it to yours
    paths.stats = '/media/joe/000FA49A000EC49F/PTB-TIR-Evaluation_toolkit/trackers/MLSSNet/TIRvideo_stats.mat'; % not used
    paths.video_base = './sequences/'; % change it to yours video path
    paths = vl_argparse(paths, varargin);
end
