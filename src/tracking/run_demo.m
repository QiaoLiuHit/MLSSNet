%run the tracker on all videos in the "sequences" folder

% set tracking sequences
seqTir={
     'airplane'  
};
 
%run tracking
for s=1:numel(seqTir)
    run_tracker(seqTir{s});
end

