root_dir="/Users/zhangjunqi/Documents/ConnectFour/"  #your path here
for file in `ls "${root_dir}Sourcecode"`
do
    cd "${root_dir}Sourcecode/${file}"
    pwd
    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang -x c++ -arch x86_64 -fmessage-length=0 -std=gnu++11 -stdlib=libc++ -Wno-trigraphs -fpascal-strings -Os -Wno-missing-field-initializers -Wno-missing-prototypes -Wreturn-type -Wno-non-virtual-dtor -Wno-overloaded-virtual -Wno-exit-time-destructors -Wformat -Wno-missing-braces -Wparentheses -Wswitch -Wno-unused-function -Wno-unused-label -Wno-unused-parameter -Wunused-variable -Wunused-value -Wempty-body -Wuninitialized -Wno-unknown-pragmas -Wno-shadow -Wno-four-char-constants -Wno-conversion -Wshorten-64-to-32 -Wno-newline-eof -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk -fasm-blocks -Wdeprecated-declarations -Winvalid-offsetof -mmacosx-version-min=10.10 -g -fvisibility-inlines-hidden -Wno-sign-conversion -I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include -I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include -I/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/include -MMD -MT dependencies -c *.cpp

    /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ *.o -arch x86_64 -dynamiclib -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.15.sdk -mmacosx-version-min=10.10 -stdlib=libc++ -single_module -compatibility_version 1 -current_version 1 -o "${file}.dylib"
    cd -
done
