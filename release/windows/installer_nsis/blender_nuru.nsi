; Blender-Nuru Windows installer (NSIS 3.x, MUI2).
;
; Build (paths may be absolute or relative to this script):
;   makensis /DAPP_VERSION=5.1.1-0.9.8 /DVI_VERSION=5.1.1.0 ^
;            /DPAYLOAD_DIR=<staged package dir> ^
;            /DASSETS_DIR=<generate_assets.py output dir> ^
;            /DOUT_FILE=<output exe path> blender_nuru.nsi
;
; The staged package dir must have the release zip layout (Blender-Nuru.exe,
; Blender-Nuru-launcher.exe, 5.1/, blender.crt/, blender.shared/, docs/, images/,
; license/, README.html). Branding assets come from generate_assets.py.

Unicode true
ManifestDPIAware true
SetCompressor /SOLID lzma

!define APP_NAME "Blender-Nuru"
!ifndef APP_VERSION
  !define APP_VERSION "5.1.1-0.9.8"
!endif
!ifndef VI_VERSION
  !define VI_VERSION "5.1.1.0"
!endif
!ifndef PAYLOAD_DIR
  !define PAYLOAD_DIR "..\..\..\builds\release-zip\${APP_NAME}-windows-${APP_VERSION}"
!endif
!ifndef ASSETS_DIR
  !define ASSETS_DIR "..\..\..\builds\release-zip\installer-assets"
!endif
!ifndef OUT_FILE
  !define OUT_FILE "..\..\..\builds\release-zip\${APP_NAME}-windows-${APP_VERSION}.exe"
!endif
!define ARP_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

Name "${APP_NAME} ${APP_VERSION}"
OutFile "${OUT_FILE}"
InstallDir "$PROGRAMFILES64\${APP_NAME}"
InstallDirRegKey HKLM "Software\${APP_NAME}" "InstallDir"
RequestExecutionLevel admin
BrandingText "Kondoo Digital GmbH - www.kondoo-digital.com"

!include "MUI2.nsh"
!include "FileFunc.nsh"

!define MUI_ICON "${ASSETS_DIR}\blender_nuru.ico"
!define MUI_UNICON "${ASSETS_DIR}\blender_nuru.ico"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "${ASSETS_DIR}\header.bmp"
!define MUI_WELCOMEFINISHPAGE_BITMAP "${ASSETS_DIR}\welcome.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "${ASSETS_DIR}\welcome.bmp"
!define MUI_ABORTWARNING

!define MUI_WELCOMEPAGE_TITLE "${APP_NAME} ${APP_VERSION}"
!define MUI_WELCOMEPAGE_TEXT "Eevee hardware ray tracing for Blender.$\r$\n$\r$\nNuru is powered by Kondoo Digital.$\r$\nwww.kondoo-digital.com$\r$\n$\r$\nThis wizard will install ${APP_NAME} ${APP_VERSION} on your computer. Click Next to continue."

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${PAYLOAD_DIR}\license\spdx\GPL-3.0-or-later.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\${APP_NAME}-launcher.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Start ${APP_NAME}"
; Desktop-shortcut checkbox on the finish page (the components-page section also offers it;
; both default on, creation is idempotent).
!define MUI_FINISHPAGE_SHOWREADME ""
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Create desktop shortcut"
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION FinishPageDesktopShortcut
!define MUI_FINISHPAGE_LINK "www.kondoo-digital.com"
!define MUI_FINISHPAGE_LINK_LOCATION "https://www.kondoo-digital.com"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

VIProductVersion "${VI_VERSION}"
VIAddVersionKey /LANG=1033 "ProductName" "${APP_NAME}"
VIAddVersionKey /LANG=1033 "ProductVersion" "${APP_VERSION}"
VIAddVersionKey /LANG=1033 "FileVersion" "${APP_VERSION}"
VIAddVersionKey /LANG=1033 "CompanyName" "Kondoo Digital GmbH"
VIAddVersionKey /LANG=1033 "FileDescription" "${APP_NAME} ${APP_VERSION} Installer"
VIAddVersionKey /LANG=1033 "LegalCopyright" "License: GNU GPL; see the license folder"

Function .onInit
  SetShellVarContext all
  ; 64-bit app: keep registry writes out of the WOW6432Node view.
  SetRegView 64
  InitPluginsDir
  File "/oname=$PLUGINSDIR\splash.bmp" "${ASSETS_DIR}\splash.bmp"
  advsplash::show 2000 0 0 0xFF00FF "$PLUGINSDIR\splash"
  Pop $0
FunctionEnd

Function FinishPageDesktopShortcut
  CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_NAME}-launcher.exe" "" "$INSTDIR\${APP_NAME}.ico" 0
FunctionEnd

Section "!${APP_NAME} (required)" SecCore
  SectionIn RO
  SetOutPath "$INSTDIR"
  File /r "${PAYLOAD_DIR}\*.*"
  File "/oname=$INSTDIR\${APP_NAME}.ico" "${ASSETS_DIR}\blender_nuru.ico"
  WriteRegStr HKLM "Software\${APP_NAME}" "InstallDir" "$INSTDIR"
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Drop stale 32-bit-view keys from installers that predate SetRegView 64.
  SetRegView 32
  DeleteRegKey HKLM "${ARP_KEY}"
  DeleteRegKey HKLM "Software\${APP_NAME}"
  SetRegView 64

  WriteRegStr HKLM "${ARP_KEY}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "${ARP_KEY}" "DisplayVersion" "${APP_VERSION}"
  WriteRegStr HKLM "${ARP_KEY}" "Publisher" "Kondoo Digital GmbH"
  WriteRegStr HKLM "${ARP_KEY}" "DisplayIcon" "$INSTDIR\${APP_NAME}.ico"
  WriteRegStr HKLM "${ARP_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${ARP_KEY}" "URLInfoAbout" "https://www.kondoo-digital.com"
  WriteRegStr HKLM "${ARP_KEY}" "HelpLink" "https://github.com/kondoodigital/blender-nuru"
  WriteRegStr HKLM "${ARP_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKLM "${ARP_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD HKLM "${ARP_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${ARP_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${ARP_KEY}" "EstimatedSize" $0
SectionEnd

Section "Start Menu shortcuts" SecStartMenu
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_NAME}-launcher.exe" "" "$INSTDIR\${APP_NAME}.ico" 0
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall ${APP_NAME}.lnk" "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Desktop shortcut" SecDesktop
  CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_NAME}-launcher.exe" "" "$INSTDIR\${APP_NAME}.ico" 0
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCore} "${APP_NAME} application files (required)."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecStartMenu} "Start Menu shortcuts for ${APP_NAME} and its uninstaller."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Desktop shortcut for ${APP_NAME}."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Function un.onInit
  SetShellVarContext all
  SetRegView 64
FunctionEnd

Section "Uninstall"
  ; Known payload only; a non-empty custom folder with user files is preserved.
  Delete "$INSTDIR\${APP_NAME}.exe"
  Delete "$INSTDIR\${APP_NAME}-launcher.exe"
  Delete "$INSTDIR\blender_cpu_check.dll"
  Delete "$INSTDIR\BlendThumb.dll"
  Delete "$INSTDIR\python3.dll"
  Delete "$INSTDIR\python313.dll"
  Delete "$INSTDIR\ucrtbase.dll"
  Delete "$INSTDIR\README.html"
  Delete "$INSTDIR\${APP_NAME}.ico"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR\5.1"
  RMDir /r "$INSTDIR\blender.crt"
  RMDir /r "$INSTDIR\blender.shared"
  RMDir /r "$INSTDIR\docs"
  RMDir /r "$INSTDIR\images"
  RMDir /r "$INSTDIR\license"
  RMDir "$INSTDIR"

  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Uninstall ${APP_NAME}.lnk"
  RMDir "$SMPROGRAMS\${APP_NAME}"
  Delete "$DESKTOP\${APP_NAME}.lnk"

  DeleteRegKey HKLM "${ARP_KEY}"
  DeleteRegKey HKLM "Software\${APP_NAME}"
SectionEnd
