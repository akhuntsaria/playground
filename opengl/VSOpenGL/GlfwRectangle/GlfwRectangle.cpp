#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1000, 800, "Shader", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }


    int nrAttributes;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &nrAttributes);
    std::cout << "Maximum nr of vertex attributes supported: " << nrAttributes << std::endl;


    unsigned int vs; // Vertex shader
    vs = glCreateShader(GL_VERTEX_SHADER);

    std::ifstream vsInFile;
    vsInFile.open("shader.vert");
    if (!vsInFile.is_open()) {
        std::cout << "Vertext shader files doesn't exist" << std::endl;
        std::cout << "Current directory: " << std::filesystem::current_path() << std::endl;
        return -1;
    }

    std::stringstream vsStream;
    vsStream << vsInFile.rdbuf();
    std::string vsString = vsStream.str();
    const char* vsSource = vsString.c_str();

    glShaderSource(vs, 1, &vsSource, NULL);
    glCompileShader(vs);

    int success;
    char log[1024];
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vs, 1024, NULL, log);
        std::cout << "Vertex shader compilation failed: " << log << std::endl;
        return -1;
    }


    unsigned int fs; // Fragment shader
    fs = glCreateShader(GL_FRAGMENT_SHADER);

    std::ifstream fsInFile;
    fsInFile.open("shader.frag");
    if (!fsInFile.is_open()) {
        std::cout << "Fragment shader files doesn't exist" << std::endl;
        return -1;
    }

    std::stringstream fsStream;
    fsStream << fsInFile.rdbuf();
    std::string fsString = fsStream.str();
    const char* fsSource = fsString.c_str();

    glShaderSource(fs, 1, &fsSource, NULL);
    glCompileShader(fs);

    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fs, 1024, NULL, log);
        std::cout << "Fragment shader compilation failed: " << log << std::endl;
        return -1;
    }


    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();

    glAttachShader(shaderProgram, vs);
    glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 1024, NULL, log);
        std::cout << "Shader program linking failed: " << log << std::endl;
        return -1;
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    float vertices[] = {
        // positions         // colors
         0.7f,  0.7f, 0.0f,  0.0f,  0.0f, 1.0f,
         0.7f, -0.7f, 0.0f,  0.0f,  1.0f, 0.0f,
        -0.7f, -0.7f, 0.0f,  1.0f,  0.0f, 0.0f,
        -0.7f,  0.7f, 0.0f,  0.0f,  1.0f, 0.0f,

        0.8f,   0.7f, 0.0f,  0.0f,  0.0f, 1.0f,
        0.8f,  -0.7f, 0.0f,  1.0f,  0.0f, 0.0f,
    };
    unsigned int indcies[] = {
        0, 1, 2,
        3, 0, 2,

        0, 4, 5,
        0, 1, 5
    };

    unsigned int EBO, VAO, VBO;
    glGenBuffers(1, &EBO);
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indcies), indcies, GL_STATIC_DRAW);
    
    // positions
    // 6 * float size = 24 = stride for this attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbinding not needed?
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    while (!glfwWindowShouldClose(window))
    {
        glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);

        float timeValue = glfwGetTime();
        float redValue = cos(timeValue) / 2.0f + 0.5f;
        int vertexColorLocation = glGetUniformLocation(shaderProgram, "ourColor");
        glUniform4f(vertexColorLocation, redValue, 1.0f, 0.0f, 1.0f);

        // Not needed since there's only one VAO?
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();

    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
