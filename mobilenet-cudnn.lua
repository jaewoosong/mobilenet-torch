local function convDepthPointCudnn(nInputPlane, nOutputPlane, stride)
    local net = nn.Sequential()
    local pad = 1
    -- (1) depthwise
    -- NOTE: Only cudnn module has 'nInputPlane' parameter in SpatialConvolution function.
    net:add(cudnn.SpatialConvolution(nInputPlane, nInputPlane, 3, 3, stride, stride, pad, pad, nInputPlane))
    net:add(cudnn.SpatialBatchNormalization(nInputPlane))
    net:add(cudnn.ReLU(true))
    -- (2) pointwise
    net:add(cudnn.SpatialConvolution(nInputPlane, nOutputPlane, 1, 1))
    net:add(cudnn.SpatialBatchNormalization(nOutputPlane))
    net:add(cudnn.ReLU(true))
    return net
end

local function mobileNet(nClasses)
    local net = nn.Sequential()

    -- (1) common convolution layer
    local ker = 3 -- kernel
    local stride = 2 -- stride
    local pad = 1 -- padding
    net:add(cudnn.SpatialConvolution(3, 32, ker, ker, stride, stride, pad, pad))
    net:add(cudnn.SpatialBatchNormalization(32))
    net:add(cudnn.ReLU(true)) -- true = in-place, false = keeping separate state.

    -- (2) depthwise followed by pointwise
    net:add(convDepthPointCudnn(32, 64, 1))
    net:add(convDepthPointCudnn(64, 128, 2))
    net:add(convDepthPointCudnn(128, 128, 1))
    net:add(convDepthPointCudnn(128, 256, 2))
    net:add(convDepthPointCudnn(256, 256, 1))
    net:add(convDepthPointCudnn(256, 512, 2))

    for i=1,5 do
        net:add(convDepthPointCudnn(512, 512, 1))
    end

    net:add(convDepthPointCudnn(512, 1024, 2))
    net:add(convDepthPointCudnn(1024, 1024, 1)) -- there is a typo in the original paper
        
    -- (3) avg. pooling and a fully-connected layer
    net:add(cudnn.SpatialAveragePooling(7,7))    
    net:add(nn.View(1024):setNumInputDims(3)) -- use Reshape (below) if there's any problem
    --net:add(nn.Reshape(1024))
    net:add(nn.Linear(1024, nClasses))
    
    return net
end
