#include "ssvio/system.hpp"

namespace ssvio {

System::System(const std::string &config_file_path)
  : sys_config_file_path_(config_file_path)
{
  LOG_ASSERT(!sys_config_file_path_.empty()) << " !sys_config_file_path_.empty() ";
  ssvio::Setting::getSingleton()->InitParamSetting(sys_config_file_path_);
  view_ui_ = std::make_shared<ui::PangolinWindow>();
  LOG_ASSERT(view_ui_->Init()) << "view_ui_->Init() Failed";
  
}

std::shared_ptr<ui::PangolinWindow> System::getViewUi() const
{
  return view_ui_;
}

} // namespace ssvio