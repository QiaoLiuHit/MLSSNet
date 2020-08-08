function opts = env_paths_training(opts)  
% for only TIR dataset
    opts.rootDataDir = '/media/joe/000A68F0000410AD/TIRDataset-MLSSNet_curated/TrainingData/'; % change  it to yours
    opts.imdbVideoPath = '/media/joe/000A68F0000410AD/scripts/TIRDataset-curation/imdb_MLSSNet-TIR.mat'; % change it to yours
    opts.imageStatsPath = '/media/joe/000FA49A000EC49F/Desktop/Joe/tracker_benchmark_v1.0/trackers/siamese_fc_master/training/imdb_video_stats.mat'; % not used 

% % for TIR+RGB dataset
%      opts.rootDataDir = '/media/joe/000A68F0000410AD/ILSVRC2015_curated/Data/VID/train/';
%      opts.imdbVideoPath = '/media/joe/000A68F0000410AD/scripts/TIRDataset-curation/imdb_MLSSNet_1_all.mat';
%      opts.imageStatsPath = '/media/joe/000FA49A000EC49F/Desktop/Joe/tracker_benchmark_v1.0/trackers/siamese_fc_master/training/imdb_video_stats.mat';
  
end

