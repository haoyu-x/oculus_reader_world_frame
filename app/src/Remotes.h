//
// Created by Jedrzej on 12/15/2020.
//

#ifndef OCULUSTELEOP_BUTTONS_H
#define OCULUSTELEOP_BUTTONS_H

#include <openxr/openxr.h>
#include "XrApp.h"
#include <string>

struct RemoteState {
    bool triggerTouchL_ = false;
    bool triggerTouchR_ = false;
    bool buttonA = false;
    bool buttonB = false;
    bool buttonX = false;
    bool buttonY = false;
    OVR::Matrix4f LeftRemoteTransform;
    OVR::Matrix4f RightRemoteTransform;
    OVR::Matrix4f HeadTransform;
};

std::string TransformationMatrixToString(const OVR::Matrix4f& transformationMatrix) {
    const size_t size = 512;
    char buffer[size];
    transformationMatrix.ToString(buffer, size);
    return std::string(buffer);
}

std::string remotes_to_string(const RemoteState& state_, const OVRFW::ovrApplFrameIn& frame_) {
        std::string text;
        if (frame_.LeftRemoteTracked) {
            text += "l:";
            text += TransformationMatrixToString(state_.LeftRemoteTransform);
            text += "|";
        }
        if (frame_.RightRemoteTracked) {
            text += "r:";
            text += TransformationMatrixToString(state_.RightRemoteTransform);
            text += "|";
        }
        // head pose in the tracking (world) frame, so that the head-relative
        // controller transforms above can be mapped back to the world frame
        text += "h:";
        text += TransformationMatrixToString(state_.HeadTransform);
        text += "&";
        if (frame_.RightRemoteTracked) {
            text += "R,"; // tell that right hand is available
            if (state_.buttonA)
                text += "A,";
            if (state_.buttonB)
                text += "B,";
            if (frame_.RightRemoteIndexTrigger > 0.25f)
                text += "RTr,";
            if (frame_.RightRemoteGripTrigger > 0.25f)
                text += "RG,";
            if (state_.triggerTouchR_)
                text += "RT,";
            text += "rightJS " + std::to_string(frame_.RightRemoteJoystick.x) + " " +
                    std::to_string(frame_.RightRemoteJoystick.y) + ",";
            text += "rightTrig " + std::to_string(frame_.RightRemoteIndexTrigger) + ",";
            text += "rightGrip " + std::to_string(frame_.RightRemoteGripTrigger);
        }
        if (frame_.RightRemoteTracked && frame_.LeftRemoteTracked) {
            text += ",";
        }
        // left hand
        if (frame_.LeftRemoteTracked) {
            text += "L,"; // tell that left hand is available
            if (state_.buttonX)
                text += "X,";
            if (state_.buttonY)
                text += "Y,";
            if (frame_.LeftRemoteIndexTrigger > 0.25f)
                text += "LTr,";
            if (frame_.LeftRemoteGripTrigger > 0.25f)
                text += "LG,";
            if (state_.triggerTouchL_)
                text += "LT,";
            text += "leftJS " + std::to_string(frame_.LeftRemoteJoystick.x) + " " +
                    std::to_string(frame_.LeftRemoteJoystick.y) + ",";
            text += "leftTrig " + std::to_string(frame_.LeftRemoteIndexTrigger) + ",";
            text += "leftGrip " + std::to_string(frame_.LeftRemoteGripTrigger);
        }
        return text;
}

#endif //OCULUSTELEOP_BUTTONS_H
