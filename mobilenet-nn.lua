require 'nn'

local function convDepthPoint(nInputPlane, nOutputPlane, stride)
    local net = nn.Sequential()
    local pad = 1
    -- (1) depthwise
    -- NOTE: SpatialDepthWiseConvolution has no cudnn equivalent
    net:add(nn.SpatialDepthWiseConvolution(nInputPlane, 1, 3, 3, stride, stride, pad, pad))
    net:add(nn.SpatialBatchNormalization(nInputPlane))
    net:add(nn.ReLU(true))
    -- (2) pointwise
    net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 1, 1))
    net:add(nn.SpatialBatchNormalization(nOutputPlane))
    net:add(nn.ReLU(true))
    return net
end

local function mobileNet(nClasses)
    local net = nn.Sequential()

    -- (1) common convolution layer
    local ker = 3 -- kernel
    local stride = 2 -- stride
    local pad = 1 -- padding
    net:add(nn.SpatialConvolution(3, 32, ker, ker, stride, stride, pad, pad))
    net:add(nn.SpatialBatchNormalization(32))
    net:add(nn.ReLU(true)) -- true = in-place, false = keeping separate state.

    -- (2) depthwise followed by pointwise
    net:add(convDepthPoint(32, 64, 1))
    net:add(convDepthPoint(64, 128, 2))
    net:add(convDepthPoint(128, 128, 1))
    net:add(convDepthPoint(128, 256, 2))
    net:add(convDepthPoint(256, 256, 1))
    net:add(convDepthPoint(256, 512, 2))

    for i=1,5 do
        net:add(convDepthPoint(512, 512, 1))
    end

    net:add(convDepthPoint(512, 1024, 2))
    net:add(convDepthPoint(1024, 1024, 1)) -- there is a typo in the original paper

    -- (3) avg. pooling and a fully-connected layer
    net:add(nn.SpatialAveragePooling(7,7))    
    net:add(nn.View(1024):setNumInputDims(3)) -- use Reshape (below) if there's any problem
    --net:add(nn.Reshape(1024))
    net:add(nn.Linear(1024, nClasses))

    return net
end
