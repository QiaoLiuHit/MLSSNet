function net = make_net_MLSSNet(opts)
% MAKE_NET constructs the network up to the output.
% Loss functions are not added.
%
% Options include:
% Architecture to use for initial branches of network.
% Whether to use scores or bounding box regression.
% Whether to use xcorr or concat layer to join.
% Architecture to put after concat (1x1 convs).

% add semantic-aware and structure-aware modules

switch opts.branch.type
    case 'alexnet'
        [branch1, branch2] = make_branches_alexnet(opts);
        
    otherwise
        error('Unknown branch type')
end
[repr_sz, repr_stride] = output_size(branch1, [opts.exemplarSize*[1 1], 3, 1]);
switch opts.join.method
    case 'xcorr'
        [join, final] = make_join_xcorr(opts);
    case 'corrfilt'
        %conv5 branch
        [join, final] = make_join_corr_filt(opts, repr_sz, repr_stride); 
        
        % conv3 branch 
        repr_sz3=[53,53,384];
        [join1, final1] = make_join_corr_filt3(opts, repr_sz3, repr_stride);

    otherwise
        error('Unknown join method')
end


% make siamese network
net = make_siamese_MLSSNet(branch1, branch2, join, join1, final, final1,...
                   {'exemplar', 'instance'}, 'score', ...
                   'share_all', opts.share_params);

end

function [branch1, branch2] = make_branches_alexnet(opts)
    branch_opts.last_layer = 'conv5';
    branch_opts.num_out     = [96, 256, 384, 384, 256];
    branch_opts.num_in      = [ 3,  48, 256, 192, 192];
    branch_opts.conv_stride = [ 2,   1,   1,   1,   1];
    branch_opts.pool_stride = [ 2,   2];
    branch_opts.batchNormalization = true;
    branch_opts = vl_argparse(branch_opts, {opts.branch.conf});

    branch_opts.exemplarSize = opts.exemplarSize * [1 1];
    branch_opts.instanceSize = opts.instanceSize * [1 1];
    branch_opts.weightInitMethod = opts.init.weightInitMethod;
    branch_opts.scale            = opts.init.scale;
    branch_opts.initBias         = opts.init.initBias;

    f = @() make_branch_alexnet(branch_opts);
    branch1 = f();
    branch2 = f();
end


function [join, final] = make_join_xcorr(opts)
    join_opts.finalBatchNorm = true;
    join_opts.adjustGainInit = 1;
    join_opts.adjustBiasInit = 0;
    % Learning rates ignored if batch-norm is enabled.
    join_opts.adjustGainLR = 0;
    join_opts.adjustBiasLR = 1;
    join_opts = vl_argparse(join_opts, {opts.join.conf});

    join = dagnn.DagNN();
    join.addLayer('xcorr', XCorr(), {'in1', 'in2'}, {'out'});

    % Create adjust layer.
    final.layers = {};
    convOpts = {'CudnnWorkspaceLimit', 1024*1024*1024};
    if join_opts.finalBatchNorm
        % Batch-norm layer only.
        final.layers{end+1} = struct(...
            'type', 'bnorm', 'name', 'adjust_bn', ...
            'weights', {{single(join_opts.adjustGainInit), ...
                         single(join_opts.adjustBiasInit), ...
                         zeros(1, 2, 'single')}}, ...
            'learningRate', [2 1 0.3], ...
            'weightDecay', [0 0]);
    else
        % Linear layer only.
        final.layers{end+1} = struct(...
            'type', 'conv', 'name', 'adjust', ...
            'weights', {{single(join_opts.adjustGainInit), ...
                         single(join_opts.adjustBiasInit)}}, ...
            'learningRate', [join_opts.adjustGainLR, join_opts.adjustBiasLR], ...
            'weightDecay', [1 0], ...
            'opts', {convOpts});
    end
end

function [join, final] = make_join_corr_filt(opts, in_sz, in_stride)
    join_opts.finalBatchNorm = false;
    join_opts.const_cf = false;
    join_opts.lambda = nan;
    join_opts.window = 'not-set';
    join_opts.window_lr = 0;
    join_opts.bias = false;
    join_opts.adjust = true;
    join_opts.sigma = 0;
    join_opts.target_lr = 0;
    
    join_opts.adjustGainInit = 1;
    join_opts.adjustBiasInit = 0;
    % Learning rates ignored if batch-norm is enabled.
    join_opts.adjustGainLR = 0;
    join_opts.adjustBiasLR = 1;
    
    join_opts = vl_argparse(join_opts, {opts.join.conf});
    convOpts = {'CudnnWorkspaceLimit', 1024*1024*1024};

    join = dagnn.DagNN();
    
    %add semantic enhancement module for conv5
    join=addSCNet(join);
    
    
    % Apply window before correlation filter.
    join.addLayer('cf_window', MulConst(), ...
                  {'br1_Wfeat5'}, {'cf_example'}, {'window'});
    p = join.getParamIndex('window');
    join.params(p).value = single(make_window(in_sz, join_opts.window));
    join.params(p).learningRate = join_opts.window_lr;

    % Establish whether there is a bias parameter to the XCorr.
    cf_outputs = {'tmpl'};
    xcorr_inputs = {'tmpl_cropped', 'in2'};

    % learnt alphas instead of CF for Adaptation Experiment
    if join_opts.const_cf
        join.addLayer('circ', ConvCircScalar(), ...
                      {'cf_example'}, cf_outputs, {'circf'});
        p = join.getParamIndex('circf');
        join.params(p).value = init_weight(opts.init, in_sz(1), in_sz(2), 1, 1, 'single');
    else
    % Add a correlation filter before the XCorr in branch 1.
        if join_opts.bias
            % Connect correlation filter bias to xcorr bias.
            cf_outputs = [cf_outputs, {'bias'}];
            xcorr_inputs = [xcorr_inputs, {'bias'}];
        end
        join.addLayer('cf', ...
                      CorrFilter('lambda', join_opts.lambda, ...
                                 'bias', join_opts.bias), ...
                      {'cf_example'}, cf_outputs, {'cf_target'});
        % Set correlation filter target.
        p = join.getParamIndex('cf_target');
    end
    
    join.addLayer('crop_z', ...
                    CropMargin('margin', 16), ...
                    cf_outputs, xcorr_inputs{1});
                
    % Cross-correlate template with features of other image.
    join.addLayer('xcorr', XCorr('bias', join_opts.bias), ...
                  xcorr_inputs, {'out'});


    assert(join_opts.sigma > 0);
    join.params(p).value = single(gaussian_response(in_sz, join_opts.sigma/in_stride));
    join.params(p).learningRate = join_opts.target_lr;

    % Add scalar layer to calibrate corr-filt scores for loss function.
    final.layers = {};
    if join_opts.adjust
        if  join_opts.finalBatchNorm
            % Batch-norm layer only.
            final.layers{end+1} = struct(...
                'type', 'bnorm', 'name', 'adjust_bn', ...
                'weights', {{single(join_opts.adjustGainInit), ...
                             single(join_opts.adjustBiasInit), ...
                             zeros(1, 2, 'single')}}, ...
                'learningRate', [2 1 0.3], ...
                'weightDecay', [0 0]);
        else
            final.layers{end+1} = struct(...
                'type', 'conv', 'name', 'adjust', ...
                'weights', {{single(1), single(-0.5)}}, ...
                'learningRate', [1, 2], ...
                'weightDecay', [0 0], ...
                'opts', {convOpts});            
        end        
    end
end


function [join3, final3] = make_join_corr_filt3(opts, in_sz, in_stride)
    join_opts.finalBatchNorm = false;
    join_opts.const_cf = false;
    join_opts.lambda = nan;
    join_opts.window = 'not-set';
    join_opts.window_lr = 0;
    join_opts.bias = false;
    join_opts.adjust = true;
    join_opts.sigma = 0;
    join_opts.target_lr = 0;
    
    join_opts.adjustGainInit = 1;
    join_opts.adjustBiasInit = 0;
    % Learning rates ignored if batch-norm is enabled.
    join_opts.adjustGainLR = 0;
    join_opts.adjustBiasLR = 1;
    
    join_opts = vl_argparse(join_opts, {opts.join.conf});
    convOpts = {'CudnnWorkspaceLimit', 1024*1024*1024};

    join3 = dagnn.DagNN();
    
     %reduce dimension to 64
    join3.addLayer('conv_dimred2', dagnn.Conv('size',[1 1 384 64],'pad',0,'stride',1,'hasBias',true), {'in1'}, {'conv3_dimred'}, {'br_conv3f_dimred','br_conv3b_dimred'});
    join3.params(join3.getParamIndex('br_conv3f_dimred')).value =init_weight(opts.init, 1, 1, 384, 64, 'single');  %--->
    join3.params(join3.getParamIndex('br_conv3b_dimred')).value=zeros(64, 1, 'single');
    
    % add structure-aware module on conv3
    join3 = addSANet(join3);
    
    % Apply window before correlation filter.
    join3.addLayer('cf_window3', MulConst(), ...
                  {'br1_Wfeat'}, {'cf_example3'}, {'window3'});
    p = join3.getParamIndex('window3');
    join3.params(p).value = single(make_window(in_sz, join_opts.window));
    join3.params(p).learningRate = join_opts.window_lr;

    % Establish whether there is a bias parameter to the XCorr.
    cf_outputs = {'tmpl3'};
    xcorr_inputs = {'tmpl_cropped3', 'in2'};

    % learnt alphas instead of CF for Adaptation Experiment
    if join_opts.const_cf
        join3.addLayer('circ3', ConvCircScalar(), ...
                      {'cf_example3'}, cf_outputs, {'circf3'});
        p = join3.getParamIndex('circf3');
        join3.params(p).value = init_weight(opts.init, in_sz(1), in_sz(2), 1, 1, 'single');
    else
    % Add a correlation filter before the XCorr in branch 1.
        if join_opts.bias
            % Connect correlation filter bias to xcorr bias.
            cf_outputs = [cf_outputs, {'bias'}];
            xcorr_inputs = [xcorr_inputs, {'bias'}];
        end
        join3.addLayer('cf3', ...
                      CorrFilter('lambda', join_opts.lambda, ...
                                 'bias', join_opts.bias), ...
                      {'cf_example3'}, cf_outputs, {'cf_target3'});
        % Set correlation filter target.
        p = join3.getParamIndex('cf_target3');
    end
    
    join3.addLayer('crop_z3', ...
                    CropMargin('margin', 16), ...
                    cf_outputs, xcorr_inputs{1});
                
    % Cross-correlate template with features of other image.
    join3.addLayer('xcorr3', XCorr('bias', join_opts.bias), ...
                  xcorr_inputs, {'out3'});


    assert(join_opts.sigma > 0);
    join3.params(p).value = single(gaussian_response(in_sz, join_opts.sigma/in_stride));
    join3.params(p).learningRate = join_opts.target_lr;

    % Add scalar layer to calibrate corr-filt scores for loss function.
    final3.layers = {};
    if join_opts.adjust
        if  join_opts.finalBatchNorm
            % Batch-norm layer only.
            final3.layers{end+1} = struct(...
                'type', 'bnorm', 'name', 'adjust_bn', ...
                'weights', {{single(join_opts.adjustGainInit), ...
                             single(join_opts.adjustBiasInit), ...
                             zeros(1, 2, 'single')}}, ...
                'learningRate', [2 1 0.3], ...
                'weightDecay', [0 0]);
        else
            final3.layers{end+1} = struct(...
                'type', 'conv', 'name', 'adjust3', ...
                'weights', {{single(1), single(-0.5)}}, ...
                'learningRate', [1, 2], ...
                'weightDecay', [0 0], ...
                'opts', {convOpts});            
        end        
    end
end



function num_out = output_dim(opts)
    % TODO: Restructure options so that this code does not need to know about
    % different types of loss function?
    switch opts.loss.type
        case {'simple', 'structured'}
            num_out = 1;
        case 'regression'
            assert(opts.instanceSize==opts.exemplarSize, 'Exemplar and Instance should have the same size.');
            assert(opts.negatives==0 && opts.hardNegatives==0, 'No negative pairs for the moment.');
            num_out = 4;
        otherwise
            error('unknown loss');
    end
end

function [out_sz, out_stride] = output_size(net, in_sz)
    % Assume that net has 1 input and 1 output.
    if isa(net, 'dagnn.DagNN')
        input = only(net.getInputs());
        output = only(net.getOutputs());
        sizes = net.getVarSizes({input, in_sz});
        out_sz = sizes{net.getVarIndex(output)}(1:3);
        rfs = net.getVarReceptiveFields(input);
        out_stride = rfs(net.getVarIndex(output)).stride;
    else
        info = vl_simplenn_display(net, 'inputSize', in_sz);
        out_sz = info.dataSize(1:3, end);
        out_stride = info.receptiveFieldStride(:, end);
    end
    out_sz = reshape(out_sz, 1, []);
    assert(all(out_stride == out_stride(1)));
    out_stride = out_stride(1);
end


function net=addSANet(net) % structue-aware module 
channel=64; r=4;
net.addLayer('br1_conv11', dagnn.Conv('size', [7,7,channel,channel/r],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [2,2]), 'conv3_dimred', 'br1_conv_11', {'br1_conv11_f', 'br1_conv11_b'});

f = net.getParamIndex('br1_conv11_f') ;
net.params(f).value=single(randn(7,7,channel,channel/r) /sqrt(1*1*channel))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('br1_conv11_b') ;
net.params(f).value=single(zeros(channel/r,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br1_SA_relu1', dagnn.ReLU(),'br1_conv_11','br1_SA_relu_1');

net.addLayer('br1_conv12', dagnn.Conv('size', [5,5,channel/r,1],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br1_SA_relu_1', 'br1_conv_12', {'br1_conv12_f', 'br1_conv12_b'});

f = net.getParamIndex('br1_conv12_f') ;
net.params(f).value=single(randn(5,5,channel/r,1) /sqrt(1*1*channel/r))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('br1_conv12_b') ;
net.params(f).value=single(zeros(1,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br1_SA_relu2', dagnn.ReLU(),'br1_conv_12','br1_SA_relu_2');

%add deconv layer for a_conv3
deconvblock=dagnn.ConvTranspose('size', [5,5,1,1], 'upsample', 1);
net.addLayer('a_deconv3',deconvblock, {'br1_SA_relu_2'},{'deconv3'},{'deconv3_f','deconv3_b'});
    
f = net.getParamIndex('deconv3_f') ;
net.params(f).value=single(randn(5,5,1,1) /sqrt(1*1*1))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('deconv3_b') ;
net.params(f).value=single(zeros(1,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br1_SA_relu3', dagnn.ReLU(),'deconv3','br1_SA_relu_3');

%add deconv2
deconvblock=dagnn.ConvTranspose('size', [7,7,1,1], 'upsample', 2);
net.addLayer('b_deconv3',deconvblock, {'br1_SA_relu_3'},{'b_deconv3'},{'b_deconv3_f','b_deconv3_b'});
    
f = net.getParamIndex('b_deconv3_f') ;
net.params(f).value=single(randn(7,7,1,1) /sqrt(1*1*1))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('b_deconv3_b') ;
net.params(f).value=single(zeros(1,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br1_sigmoid1', dagnn.Sigmoid(),'b_deconv3','br1_sigmoid_1');
net.addLayer('br1_scale1',dagnn.Scale('hasBias',0),{'conv3_dimred','br1_sigmoid_1'},'br1_Wfeat')
end

function net=addSCNet(net) % semantic-aware module
channel=64; r=4;
%GAP branch
net.addLayer('br1_GAP', dagnn.GlobalPooling('method','avg'),'in1','GAP_out');
net.addLayer('br1_conv51', dagnn.Conv('size', [1,1,channel,channel/r],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'GAP_out', 'br1_conv_51', {'br1_conv51_f', 'br1_conv51_b'});

f = net.getParamIndex('br1_conv51_f') ;
net.params(f).value=single(randn(1,1,channel,channel/r) /sqrt(1*1*channel))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('br1_conv51_b') ;
net.params(f).value=single(zeros(channel/r,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

net.addLayer('br1_SC_relu1', dagnn.ReLU(),'br1_conv_51','br1_SC_relu_1');

net.addLayer('br1_conv52', dagnn.Conv('size', [1,1,channel/r,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br1_SC_relu_1', 'br1_conv_52', {'br1_conv52_f', 'br1_conv52_b'});

f = net.getParamIndex('br1_conv52_f') ;
net.params(f).value=single(randn(1,1,channel/r,channel) /sqrt(1*1*channel/r))/1e8;
net.params(f).learningRate=1;
net.params(f).weightDecay=10;

f = net.getParamIndex('br1_conv52_b') ;
net.params(f).value=single(zeros(channel,1));
net.params(f).learningRate=2;
net.params(f).weightDecay=10;

%GMP branch  shared conv parameters with GAP branch
net.addLayer('br1_GMP', dagnn.GlobalPooling('method','max'),'in1','GMP_out');
net.addLayer('br1_conv53', dagnn.Conv('size', [1,1,channel,channel/r],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'GMP_out', 'br1_conv_53', {'br1_conv51_f', 'br1_conv51_b'});

net.addLayer('br1_SC_relu3', dagnn.ReLU(),'br1_conv_53','br1_SC_relu_3');

net.addLayer('br1_conv54', dagnn.Conv('size', [1,1,channel/r,channel],...
    'hasBias', true, 'pad',...
    [0,0,0,0], 'stride', [1,1]), 'br1_SC_relu_3', 'br1_conv_54', {'br1_conv52_f', 'br1_conv52_b'});

net.addLayer('br1_Sum', dagnn.Sum(),{'br1_conv_52','br1_conv_54'},'Sum_out');

%add sigmoid
net.addLayer('br1_sigmoid2', dagnn.Sigmoid(),'Sum_out','br1_sigmoid_2');
net.addLayer('br1_scale2',dagnn.Scale('hasBias',0),{'in1','br1_sigmoid_2'},'br1_Wfeat5')
end