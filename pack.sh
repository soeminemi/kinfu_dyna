#!/bin/bash

# 创建目录结构
mkdir -p kinfu-dyna/DEBIAN
mkdir -p kinfu-dyna/usr/local/bin
mkdir -p kinfu-dyna/usr/share/kinfu-dyna
mkdir -p kinfu-dyna/usr/lib/kinfu-dyna
# 复制文件
cp build/bin/demo kinfu-dyna/usr/local/bin/kinfu-dyna
cp gettestdata.py piliang.py run.py testServer.py kinfu-dyna/usr/share/kinfu-dyna/
cp -r data kinfu-dyna/usr/share/kinfu-dyna/
cp -r build/lib/* kinfu-dyna/usr/lib/kinfu-dyna/

# 创建 control 文件
cat > kinfu-dyna/DEBIAN/control << EOF
Package: kinfu-dyna
Version: 1.0
Section: utils
Priority: optional
Architecture: amd64
Maintainer: Your Name <your.email@example.com>
Description: KinFu Dyna package
 This package contains KinFu Dyna software.
EOF

# 设置正确的权限
sudo chown -R root:root kinfu-dyna
sudo chmod -R 755 kinfu-dyna
sudo chmod 644 kinfu-dyna/DEBIAN/control

# 创建 .deb 包
dpkg-deb --build kinfu-dyna

echo "Package created: kinfu-dyna.deb"