FROM ros:jazzy
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-opencv \
    python3-numpy \
    python3-pip \
    git \
    build-essential \
    libusb-1.0-0-dev \
    wget \
    doxygen \
    joystick \
    nano \
    ros-jazzy-joy \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    ros-jazzy-teleop-twist-joy \
    ros-jazzy-sensor-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-nav-msgs \
    ros-jazzy-nav2-map-server \
    ros-jazzy-slam-toolbox \
    ros-jazzy-tf2-tools \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2 \
    ros-jazzy-phidgets-spatial \
    ros-jazzy-sick-scan-xd \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-pcl-conversions \
    ros-jazzy-pcl-ros \
    ros-jazzy-nav2-map-server \
    ros2-testing-apt-source \
    && apt-get update && apt-get install -y \
    ros-jazzy-depthai-ros-v3 \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN rosdep update

# ── Python dependencies ────────────────────────────────────────────────────
# Install in one layer, ordered from largest to smallest download.
# numpy<2 is pinned because ultralytics + torch can break with numpy 2.x.
RUN pip3 install --break-system-packages \
    "numpy<2" \
    torch \
    torchvision \
    ultralytics \
    depthai

# Build AriaCoda
WORKDIR /tmp
RUN git clone https://github.com/reedhedges/AriaCoda.git
WORKDIR /tmp/AriaCoda
RUN make -j$(nproc)
RUN make install

ENV ARIA_DIR=/usr/local/Aria
ENV LD_LIBRARY_PATH=/usr/local/lib:${ARIA_DIR}/lib:$LD_LIBRARY_PATH
ENV PATH=${ARIA_DIR}/bin:$PATH

# Build Lakibeam driver
RUN git clone https://github.com/RichbeamTechnology/Lakibeam_ROS2_Driver.git \
    /opt/lakibeam_ws/src/lakibeam1
WORKDIR /opt/lakibeam_ws
RUN . /opt/ros/jazzy/setup.sh && \
    colcon build --symlink-install
RUN echo "source /opt/lakibeam_ws/install/setup.bash" >> /root/.bashrc

# Build main workspace
WORKDIR /root/ros2_autobot
COPY src ./src
COPY launch ./launch

RUN . /opt/ros/jazzy/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y

RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    source /opt/lakibeam_ws/install/setup.bash && \
    colcon build --symlink-install"

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo "source /root/ros2_autobot/install/setup.bash" >> /root/.bashrc

COPY launch /root/ros2_autobot/launch

CMD ["bash"]