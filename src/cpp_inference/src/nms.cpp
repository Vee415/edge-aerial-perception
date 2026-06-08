// nms.cpp is now integrated into trt_engine.cpp
// The decode_output(), nms(), and compute_iou() functions are methods
// of TRTEngine, keeping everything together for simplicity.
//
// This file exists only so the CMakeLists.txt source list remains valid.
// If you prefer a standalone NMS implementation, you can move the relevant
// code here and have TRTEngine call it.