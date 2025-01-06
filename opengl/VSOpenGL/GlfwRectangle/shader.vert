#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 ourColor;

void main()
{
    gl_Position = vec4(aPos.x + 0.1, aPos.y - 0.1, aPos.z + 0.1, 1.0);
    ourColor = aColor;
}
